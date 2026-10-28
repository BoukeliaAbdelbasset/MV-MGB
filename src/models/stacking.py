"""
Multi-View Stacking Engine (Tier-1 Base Models + Tier-2 Meta-Learner).
Supports out-of-fold probability simplex generation, refit vs. fold-averaging,
raw feature appending, and calibrated probability mapping.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ..views.view1_tfidf import make_tfidf_pipeline


def make_tier1_estimators(random_state: int = 42, class_weight: str = None) -> dict:
    """Instantiate the default Tier-1 base pipelines for each view."""
    return {
        'view1_tfidf': make_tfidf_pipeline(
            max_features=35_000,
            ngram_range=(3, 8),
            C=1.0,
            random_state=random_state
        ),
        'view2_nv': Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.1,
                class_weight=class_weight,
                random_state=random_state
            ))
        ]),
        'view3_bsf': Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', RandomForestClassifier(
                n_estimators=200,
                max_features='sqrt',
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
        'view4_bsp': Pipeline([
            ('scaler', StandardScaler()),
            ('estimator', MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                early_stopping=True,
                random_state=random_state
            ))
        ])
    }


class MultiViewStacking(BaseEstimator, ClassifierMixin):
    """
    Two-Tier Multi-View Stacked Generalization Engine.
    
    Parameters:
      tier1_estimators: dict of name -> Pipeline/Classifier for each view.
      meta_learner: Tier-2 classifier (defaults to HistGradientBoostingClassifier).
      n_folds: Number of stratified cross-validation folds for OOF generation.
      test_inference_mode: 'refit' (train on full training set) or 'fold_average' (average fold models).
      append_raw_features: If True, concatenate raw dense physical features to meta-features.
      random_state: Random seed for CV splitting and estimators.
    """
    def __init__(
        self,
        tier1_estimators: dict = None,
        meta_learner: BaseEstimator = None,
        n_folds: int = 5,
        test_inference_mode: str = 'refit',
        append_raw_features: bool = False,
        random_state: int = 42
    ):
        self.tier1_estimators = tier1_estimators
        self.meta_learner = meta_learner
        self.n_folds = n_folds
        self.test_inference_mode = test_inference_mode
        self.append_raw_features = append_raw_features
        self.random_state = random_state

    def fit(self, view_data: dict, y: np.ndarray):
        """
        Fit Tier-1 models across stratified folds to generate out-of-fold
        probability simplices, then train the Tier-2 meta-learner.
        
        view_data: dict mapping view name to training data array/list.
          e.g. {'view1_tfidf': seq_list, 'view2_nv': X_nv, 'view3_bsf': X_bsf, 'view4_bsp': X_bsp}
        y: 1D array of class labels.
        """
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        K = len(self.classes_)
        N = len(y_encoded)
        
        if self.tier1_estimators is None:
            self.tier1_estimators = make_tier1_estimators(self.random_state)
            
        self.view_names_ = sorted(list(self.tier1_estimators.keys()))
        
        # Meta-learner setup
        if self.meta_learner is None:
            self.meta_learner_ = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_depth=8,
                l2_regularization=0.1,
                random_state=self.random_state
            )
        else:
            self.meta_learner_ = clone(self.meta_learner)

        # Cross-validation setup
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.cv_splits_ = list(skf.split(np.zeros(N), y_encoded))
        
        # Containers for fold models and OOF probability matrices
        self.fold_models_ = {v: [] for v in self.view_names_}
        self.refit_models_ = {}
        oof_probs = {v: np.zeros((N, K), dtype=np.float64) for v in self.view_names_}
        
        # Track fit spy records for verification tests
        self.fit_call_records_ = {v: [] for v in self.view_names_}

        # Step 1: In-fold training and OOF probability generation
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splits_):
            y_fold_tr = y_encoded[train_idx]
            y_fold_val = y_encoded[val_idx]
            
            for v in self.view_names_:
                X_v = view_data[v]
                # Slice input appropriately for list (strings) or numpy array
                if isinstance(X_v, list):
                    X_tr_fold = [X_v[i] for i in train_idx]
                    X_val_fold = [X_v[i] for i in val_idx]
                else:
                    X_tr_fold = X_v[train_idx]
                    X_val_fold = X_v[val_idx]
                    
                model = clone(self.tier1_estimators[v])
                model.fit(X_tr_fold, y_fold_tr)
                self.fold_models_[v].append(model)
                self.fit_call_records_[v].append({'fold': fold_idx, 'n_train': len(train_idx), 'n_val': len(val_idx)})
                
                # Predict probabilities on holdout fold
                val_proba = model.predict_proba(X_val_fold)
                
                # Align probabilities in case of missing classes in a fold
                model_classes = model.classes_
                if len(model_classes) == K:
                    oof_probs[v][val_idx] = val_proba
                else:
                    for c_idx in range(K):
                        if c_idx in model_classes:
                            pos = np.where(model_classes == c_idx)[0][0]
                            oof_probs[v][val_idx, c_idx] = val_proba[:, pos]

        # Step 2: Full refit models if test_inference_mode == 'refit'
        for v in self.view_names_:
            refit_model = clone(self.tier1_estimators[v])
            refit_model.fit(view_data[v], y_encoded)
            self.refit_models_[v] = refit_model

        # Step 3: Construct unified meta-feature tensor
        # Z_OOF: Concatenate probability simplices [P_1, P_2, P_3, P_4] -> (N, 4 * K)
        meta_blocks = [oof_probs[v] for v in self.view_names_]
        
        if self.append_raw_features:
            dense_blocks = []
            for v in self.view_names_:
                if v != 'view1_tfidf':
                    dense_blocks.append(view_data[v])
            if dense_blocks:
                meta_blocks.extend(dense_blocks)
                
        self.oof_meta_features_ = np.hstack(meta_blocks)
        self.meta_learner_.fit(self.oof_meta_features_, y_encoded)
        return self

    def _get_tier1_test_meta_features(self, view_data: dict) -> np.ndarray:
        """Compute Tier-1 probability simplex for test sequences."""
        K = len(self.classes_)
        meta_blocks = []
        
        for v in self.view_names_:
            X_v = view_data[v]
            N_test = len(X_v)
            
            if self.test_inference_mode == 'refit':
                model = self.refit_models_[v]
                proba = model.predict_proba(X_v)
                if len(model.classes_) == K:
                    meta_blocks.append(proba)
                else:
                    aligned = np.zeros((N_test, K), dtype=np.float64)
                    for c_idx in range(K):
                        if c_idx in model.classes_:
                            pos = np.where(model.classes_ == c_idx)[0][0]
                            aligned[:, c_idx] = proba[:, pos]
                    meta_blocks.append(aligned)
                
            elif self.test_inference_mode == 'fold_average':
                fold_probas = []
                for model in self.fold_models_[v]:
                    proba = model.predict_proba(X_v)
                    if len(model.classes_) == K:
                        fold_probas.append(proba)
                    else:
                        aligned = np.zeros((N_test, K), dtype=np.float64)
                        for c_idx in range(K):
                            if c_idx in model.classes_:
                                pos = np.where(model.classes_ == c_idx)[0][0]
                                aligned[:, c_idx] = proba[:, pos]
                        fold_probas.append(aligned)
                meta_blocks.append(np.mean(fold_probas, axis=0))
            else:
                raise ValueError(f"Unknown test_inference_mode: {self.test_inference_mode}")

        if self.append_raw_features:
            dense_blocks = []
            for v in self.view_names_:
                if v != 'view1_tfidf':
                    dense_blocks.append(view_data[v])
            if dense_blocks:
                meta_blocks.extend(dense_blocks)

        return np.hstack(meta_blocks)

    def predict_proba(self, view_data: dict) -> np.ndarray:
        """Compute final calibrated class probabilities via Tier-2 meta-learner."""
        X_meta = self._get_tier1_test_meta_features(view_data)
        return self.meta_learner_.predict_proba(X_meta)

    def predict(self, view_data: dict) -> np.ndarray:
        """Predict class labels with maximum posterior probability."""
        probas = self.predict_proba(view_data)
        max_idx = np.argmax(probas, axis=1)
        return self.label_encoder_.inverse_transform(max_idx)
