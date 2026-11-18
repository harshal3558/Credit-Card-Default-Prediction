import os
import sys
import io
import base64
import numpy as np
import pandas as pd

# Set matplotlib backend to Agg for non-interactive server rendering
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.CCDP.logger import logging
from src.CCDP.exception import CustomException

# Baseline population statistics from historical training data
BASELINE_STATS = {
    'non_default': {
        'LIMIT_BAL': 177305.0,
        'PAY_0': -0.21,
        'AVG_PAY_DELAY': -0.35,
        'CREDIT_UTILIZATION': 0.40,
        'PAY_AMT1': 6290.0,
    },
    'default': {
        'LIMIT_BAL': 129547.0,
        'PAY_0': 0.67,
        'AVG_PAY_DELAY': 0.16,
        'CREDIT_UTILIZATION': 0.49,
        'PAY_AMT1': 3497.0,
    }
}

FEATURE_METADATA = {
    'LIMIT_BAL': {'name': 'Credit Limit', 'category': 'profile'},
    'SEX': {'name': 'Gender', 'category': 'demographic'},
    'EDUCATION': {'name': 'Education Level', 'category': 'demographic'},
    'MARRIAGE': {'name': 'Marital Status', 'category': 'demographic'},
    'AGE': {'name': 'Age', 'category': 'demographic'},
    'PAY_0': {'name': 'Repayment Status (Sept)', 'category': 'delay'},
    'PAY_2': {'name': 'Repayment Status (Aug)', 'category': 'delay'},
    'PAY_3': {'name': 'Repayment Status (July)', 'category': 'delay'},
    'PAY_4': {'name': 'Repayment Status (June)', 'category': 'delay'},
    'PAY_5': {'name': 'Repayment Status (May)', 'category': 'delay'},
    'PAY_6': {'name': 'Repayment Status (Apr)', 'category': 'delay'},
    'BILL_AMT1': {'name': 'Bill Statement (Sept)', 'category': 'bill'},
    'BILL_AMT2': {'name': 'Bill Statement (Aug)', 'category': 'bill'},
    'PAY_AMT1': {'name': 'Payment Amount (Sept)', 'category': 'payment'},
    'PAY_AMT2': {'name': 'Payment Amount (Aug)', 'category': 'payment'},
    'AVG_PAY_DELAY': {'name': 'Average Repayment Delay', 'category': 'delay'},
    'MAX_PAY_DELAY': {'name': 'Maximum Repayment Delay', 'category': 'delay'},
    'NUM_MONTHS_DELAYED': {'name': 'Months with Payment Delay', 'category': 'delay'},
    'RECENCY_WEIGHTED_DELAY': {'name': 'Recency-Weighted Payment Delay', 'category': 'delay'},
    'IS_CONSECUTIVE_DELAY': {'name': 'Consecutive Late Payments (Sept-Aug)', 'category': 'delay'},
    'TOTAL_BILL_AMT': {'name': 'Total 6-Month Billed Amount', 'category': 'bill'},
    'AVG_BILL_AMT': {'name': 'Average Monthly Bill', 'category': 'bill'},
    'BILL_TREND': {'name': 'Bill Growth Trend', 'category': 'bill'},
    'TOTAL_PAY_AMT': {'name': 'Total 6-Month Paid Amount', 'category': 'payment'},
    'AVG_PAY_AMT': {'name': 'Average Monthly Payment', 'category': 'payment'},
    'PAY_TREND': {'name': 'Payment Trend', 'category': 'payment'},
    'CREDIT_UTILIZATION': {'name': 'Credit Utilization Ratio', 'category': 'ratio'},
    'CREDIT_UTILIZATION_RECENT': {'name': 'Recent Credit Utilization (Sept)', 'category': 'ratio'},
    'AVG_CREDIT_UTILIZATION': {'name': 'Average Credit Utilization', 'category': 'ratio'},
    'PAY_TO_BILL_RATIO': {'name': 'Payment-to-Bill Ratio', 'category': 'ratio'},
    'RECENT_PAY_FULFILLMENT': {'name': 'Recent Bill Fulfillment Rate', 'category': 'ratio'},
    'NET_UNPAID_BALANCE': {'name': 'Net Cumulative Unpaid Balance', 'category': 'debt'},
    'IS_YOUTH': {'name': 'Young Borrower Flag (<30)', 'category': 'demographic'},
    'AGE_GROUP': {'name': 'Age Group Category', 'category': 'demographic'},
}


class ModelExplainer:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = self._extract_feature_names()

    def _extract_feature_names(self):
        try:
            if hasattr(self.preprocessor, "named_steps") and "feature_eng" in self.preprocessor.named_steps:
                fe = self.preprocessor.named_steps["feature_eng"]
                if hasattr(fe, "expected_columns") and fe.expected_columns:
                    return fe.expected_columns
            if hasattr(self.model, "feature_names_in_") and self.model.feature_names_in_ is not None:
                return list(self.model.feature_names_in_)
        except Exception:
            pass
        return [f"Feature_{i}" for i in range(getattr(self.model, "n_features_in_", 34))]

    def _compute_attributions(self, X_transformed):
        """
        Computes exact local feature attributions using tree path decomposition
        for Gradient Boosting and Random Forest, or linear coefficients for linear models.
        """
        X_arr = np.asarray(X_transformed, dtype=np.float32)
        n_features = X_arr.shape[1]
        contributions = np.zeros(n_features, dtype=float)
        base_value = 0.0

        # Case 1: Gradient Boosting Classifier
        if hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0 and hasattr(self.model.estimators_[0], "__len__"):
            raw_init = self.model._raw_predict_init(X_arr)[0, 0]
            lr = self.model.learning_rate
            base_value = raw_init

            for stage in self.model.estimators_:
                tree = stage[0].tree_
                node_indicator = tree.decision_path(X_arr).toarray()[0]
                nodes = np.where(node_indicator == 1)[0]
                base_value += lr * tree.value[0, 0, 0]
                for k in range(len(nodes) - 1):
                    p, c = nodes[k], nodes[k + 1]
                    feat = tree.feature[p]
                    if feat >= 0:
                        contributions[feat] += lr * (tree.value[c, 0, 0] - tree.value[p, 0, 0])

        # Case 2: Random Forest Classifier
        elif hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0 and hasattr(self.model.estimators_[0], "tree_"):
            n_estimators = len(self.model.estimators_)
            base_vals = []
            for dt in self.model.estimators_:
                tree = dt.tree_
                node_indicator = tree.decision_path(X_arr).toarray()[0]
                nodes = np.where(node_indicator == 1)[0]
                node_sums = tree.value[:, 0].sum(axis=1)
                node_probs = tree.value[:, 0, 1] / np.maximum(node_sums, 1e-9)
                base_vals.append(node_probs[0])
                for k in range(len(nodes) - 1):
                    p, c = nodes[k], nodes[k + 1]
                    feat = tree.feature[p]
                    if feat >= 0:
                        contributions[feat] += (node_probs[c] - node_probs[p]) / n_estimators
            base_value = float(np.mean(base_vals))

        # Case 3: Logistic Regression / Linear Models
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_[0]
            contributions = coef * X_arr[0]
            base_value = float(self.model.intercept_[0]) if hasattr(self.model, "intercept_") else 0.0

        # Fallback: Feature importances heuristic
        else:
            importances = getattr(self.model, "feature_importances_", np.ones(n_features) / n_features)
            contributions = (X_arr[0] - np.mean(X_arr[0])) * importances
            base_value = 0.5

        return base_value, contributions

    def _format_value(self, feature, raw_df, engineered_df):
        """Formats the feature value into an intuitive, human-friendly representation."""
        if feature in raw_df.columns:
            val = raw_df[feature].iloc[0]
        elif feature in engineered_df.columns:
            val = engineered_df[feature].iloc[0]
        else:
            return "N/A"

        if feature == "LIMIT_BAL":
            return f"NT$ {val:,.0f}"
        elif feature in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                         "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                         "TOTAL_BILL_AMT", "AVG_BILL_AMT", "TOTAL_PAY_AMT", "AVG_PAY_AMT", "NET_UNPAID_BALANCE"]:
            return f"NT$ {val:,.0f}"
        elif feature.startswith("PAY_") and len(feature) == 5:
            if val <= -1:
                return "Paid in Full / Ahead"
            elif val == 0:
                return "Revolving Credit / On-Time"
            else:
                return f"{int(val)} Month(s) Overdue"
        elif feature in ["CREDIT_UTILIZATION", "CREDIT_UTILIZATION_RECENT", "AVG_CREDIT_UTILIZATION"]:
            return f"{val * 100:.1f}% of Limit"
        elif feature in ["PAY_TO_BILL_RATIO", "RECENT_PAY_FULFILLMENT"]:
            return f"{val * 100:.1f}% Repaid"
        elif feature == "AVG_PAY_DELAY":
            return f"{val:.2f} Months Avg"
        elif feature == "MAX_PAY_DELAY":
            return f"{val:.0f} Months Max"
        elif feature == "NUM_MONTHS_DELAYED":
            return f"{int(val)} Month(s)"
        elif feature == "IS_CONSECUTIVE_DELAY":
            return "Yes (Overdue in Sept & Aug)" if val == 1 else "No"
        elif feature == "EDUCATION":
            edu_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
            return edu_map.get(val, "Other")
        elif feature == "MARRIAGE":
            m_map = {1: "Married", 2: "Single", 3: "Other"}
            return m_map.get(val, "Single")
        elif feature == "SEX":
            return "Male" if val == 1 else "Female"
        elif feature == "AGE":
            return f"{int(val)} years old"
        elif isinstance(val, (int, np.integer)):
            return str(val)
        elif isinstance(val, (float, np.floating)):
            return f"{val:.2f}"
        return str(val)

    def _generate_feature_explanation(self, feature, val_str, impact, is_positive_driver):
        """Generates plain English reason context for why this feature acted as risk or protective."""
        name = FEATURE_METADATA.get(feature, {}).get('name', feature)

        if is_positive_driver:
            # Pushing towards DEFAULT
            if feature.startswith("PAY_"):
                return f"{name} shows overdue payment history ({val_str}), indicating severe debt servicing friction."
            if "UTILIZATION" in feature:
                return f"High credit utilization ({val_str}) places the borrower near credit limit exhaustion."
            if feature in ["PAY_TO_BILL_RATIO", "RECENT_PAY_FULFILLMENT"]:
                return f"Payment adequacy is low ({val_str}), leaving accumulating unpaid card balances."
            if feature == "LIMIT_BAL":
                return f"Lower credit line ({val_str}) provides minimal financial liquidity cushion."
            if feature == "IS_CONSECUTIVE_DELAY":
                return "Consecutive monthly delays signal ongoing chronic delinquency."
            if feature == "NET_UNPAID_BALANCE":
                return f"Cumulative unpaid debt ({val_str}) substantially escalates default exposure."
            if feature == "NUM_MONTHS_DELAYED":
                return f"Multiple overdue billing cycles ({val_str}) confirm repeated repayment failure."
            return f"{name} ({val_str}) shifted the risk profile unfavorably (+{impact:.2f} impact)."
        else:
            # Pushing towards NON-DEFAULT (Protective)
            if feature.startswith("PAY_"):
                return f"{name} reflects timely repayment ({val_str}), demonstrating solid discipline."
            if "UTILIZATION" in feature:
                return f"Controlled credit usage ({val_str}) keeps borrowing safely below maximum capacity."
            if feature in ["PAY_TO_BILL_RATIO", "RECENT_PAY_FULFILLMENT"]:
                return f"High repayment fulfillment ({val_str}) demonstrates ability to settle obligations."
            if feature == "LIMIT_BAL":
                return f"Generous credit limit ({val_str}) signals established bank trust and financial standing."
            if feature in ["MAX_PAY_DELAY", "AVG_PAY_DELAY"]:
                return f"Minimal or zero payment delay ({val_str}) confirms consistent debt servicing."
            if feature == "IS_CONSECUTIVE_DELAY":
                return "Absence of consecutive payment delays confirms stable account management."
            return f"{name} ({val_str}) acted as a protective factor (-{abs(impact):.2f} impact)."

    def explain(self, raw_input_df):
        """
        Full prediction explainability routine:
        - Runs preprocessor & model
        - Decomposes exact local attributions
        - Formulates human reasons for positive and negative signals
        - Generates 3 visual proof charts
        - Constructs executive summary
        """
        try:
            # Transform data through pipeline
            engineered_df = raw_input_df.copy()
            if hasattr(self.preprocessor, "named_steps") and "feature_eng" in self.preprocessor.named_steps:
                fe = self.preprocessor.named_steps["feature_eng"]
                engineered_df = fe.transform(engineered_df)

            X_transformed = self.preprocessor.transform(raw_input_df)

            # Model prediction and calibrated probabilities
            if hasattr(self.model, "predict_proba"):
                prob_default = float(self.model.predict_proba(X_transformed)[0, 1])
            else:
                raw_pred = float(self.model.predict(X_transformed)[0])
                prob_default = 1.0 if raw_pred == 1 else 0.0

            threshold = float(getattr(self.model, "custom_threshold_", 0.30))
            is_default = 1 if prob_default >= threshold else 0

            # Compute local attributions
            base_val, contribs = self._compute_attributions(X_transformed)

            feature_contributions = []
            for feat, cont in zip(self.feature_names, contribs):
                val_str = self._format_value(feat, raw_input_df, engineered_df)
                human_name = FEATURE_METADATA.get(feat, {}).get('name', feat)
                feature_contributions.append({
                    'feature': feat,
                    'name': human_name,
                    'value': val_str,
                    'contribution': float(cont),
                    'abs_contribution': abs(float(cont))
                })

            df_contribs = pd.DataFrame(feature_contributions)

            # Separate Positive (Risk Drivers) and Negative (Protective Factors)
            positive_df = df_contribs[df_contribs['contribution'] > 0.001].sort_values(
                by='contribution', ascending=False
            )
            negative_df = df_contribs[df_contribs['contribution'] < -0.001].sort_values(
                by='contribution', ascending=True
            )

            positive_reasons = []
            for _, row in positive_df.head(5).iterrows():
                positive_reasons.append({
                    'name': row['name'],
                    'feature': row['feature'],
                    'value': row['value'],
                    'impact': round(row['contribution'], 3),
                    'description': self._generate_feature_explanation(row['feature'], row['value'], row['contribution'], True)
                })

            negative_reasons = []
            for _, row in negative_df.head(5).iterrows():
                negative_reasons.append({
                    'name': row['name'],
                    'feature': row['feature'],
                    'value': row['value'],
                    'impact': round(row['contribution'], 3),
                    'description': self._generate_feature_explanation(row['feature'], row['value'], row['contribution'], False)
                })

            # Risk tier
            if prob_default >= 0.60:
                risk_tier = "High Risk"
            elif prob_default >= threshold:
                risk_tier = "Moderate Risk (Default Zone)"
            elif prob_default >= 0.20:
                risk_tier = "Low-to-Moderate (Safe Zone)"
            else:
                risk_tier = "Very Low Risk (Safe)"

            # Executive summary
            pos_names = [r['name'] for r in positive_reasons[:2]]
            neg_names = [r['name'] for r in negative_reasons[:2]]
            
            if is_default == 1:
                summary = (
                    f"The model predicted that this cardholder is LIKELY TO DEFAULT with an estimated default probability "
                    f"of {prob_default * 100:.1f}%, exceeding the decision threshold of {threshold * 100:.1f}%. "
                    f"The primary adverse factors driving this high-risk classification were "
                    f"'{', '.join(pos_names) if pos_names else 'delinquency indicators'}'. "
                    f"While positive factors like '{', '.join(neg_names) if neg_names else 'profile stability'}' provided minor mitigation, "
                    f"they were insufficient to offset the elevated risk of non-payment."
                )
            else:
                summary = (
                    f"The model predicted that this cardholder is NOT LIKELY TO DEFAULT with an estimated default probability "
                    f"of {prob_default * 100:.1f}%, remaining safely below the decision threshold of {threshold * 100:.1f}%. "
                    f"The favorable assessment is predominantly driven by protective strengths including "
                    f"'{', '.join(neg_names) if neg_names else 'on-time payment behavior and low utilization'}'. "
                    f"Minor risk indicators '{', '.join(pos_names) if pos_names else 'normal account fluctuations'}' remained well within healthy risk tolerance bounds."
                )

            # Generate Proof Plots
            plot_impact = self._plot_feature_impact(positive_df, negative_df)
            plot_gauge = self._plot_risk_gauge(prob_default, threshold)
            plot_benchmark = self._plot_benchmark(raw_input_df, engineered_df)

            verdict_text = "The cardholder is likely to DEFAULT." if is_default == 1 else "The cardholder is NOT likely to default."

            return {
                'prediction': is_default,
                'prediction_text': verdict_text,
                'default_probability': round(prob_default * 100, 1),
                'safety_probability': round((1.0 - prob_default) * 100, 1),
                'threshold': round(threshold * 100, 1),
                'risk_tier': risk_tier,
                'executive_summary': summary,
                'positive_reasons': positive_reasons,
                'negative_reasons': negative_reasons,
                'plot_feature_impact': plot_impact,
                'plot_risk_gauge': plot_gauge,
                'plot_benchmark': plot_benchmark
            }

        except Exception as e:
            logging.error(f"Error in ModelExplainer.explain: {e}")
            raise CustomException(e, sys)

    def _plot_feature_impact(self, positive_df, negative_df):
        """
        Proof Plot 1: Diverging horizontal bar chart showing Top Risk Drivers vs Protective Factors.
        """
        try:
            top_pos = positive_df.head(5).copy()
            top_neg = negative_df.head(5).copy()

            combined = pd.concat([top_neg.iloc[::-1], top_pos])
            if combined.empty:
                return None

            fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
            ax.set_facecolor('#1e293b')

            y_pos = np.arange(len(combined))
            values = combined['contribution'].values
            names = [f"{row['name']} ({row['value']})" for _, row in combined.iterrows()]
            colors = ['#ef4444' if v > 0 else '#10b981' for v in values]

            bars = ax.barh(y_pos, values, color=colors, height=0.6, edgecolor='none', alpha=0.9)

            # Zero reference line
            ax.axvline(0, color='#64748b', linestyle='--', linewidth=1.2, alpha=0.7)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, color='#f8fafc', fontsize=9.5, fontweight='bold')
            ax.tick_params(axis='x', colors='#94a3b8')

            # Value labels on bars
            for bar, val in zip(bars, values):
                x_val = bar.get_width()
                offset = 0.015 if x_val >= 0 else -0.015
                ha = 'left' if x_val >= 0 else 'right'
                sign = '+' if x_val > 0 else ''
                ax.text(x_val + offset, bar.get_y() + bar.get_height() / 2,
                        f"{sign}{x_val:.3f}",
                        va='center', ha=ha,
                        color='#f8fafc', fontsize=9, fontweight='bold')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')

            ax.set_title('Proof 1: Feature Impact Breakdown (Why the Model Decided This)',
                         color='#f8fafc', fontsize=12.5, fontweight='bold', pad=15)
            ax.set_xlabel('Contribution to Default Risk Score (Red = Increases Default Risk | Green = Protects Against Default)',
                          color='#94a3b8', fontsize=9.5, labelpad=10)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error plotting feature impact: {e}")
            return None

    def _plot_risk_gauge(self, prob_default, threshold):
        """
        Proof Plot 2: Visual Gauge comparing Default Probability against Calibrated Decision Threshold.
        """
        try:
            fig, ax = plt.subplots(figsize=(9, 3.2), facecolor='#0f172a')
            ax.set_facecolor('#0f172a')

            # Horizontal Gauge Meter from 0% to 100%
            bar_y = 0.5
            bar_height = 0.35

            # Background zones: Green (Safe), Amber (Moderate), Red (High Default)
            ax.barh(bar_y, threshold * 100, left=0, height=bar_height, color='#059669', alpha=0.35, label='Safe Zone (Non-Default)')
            ax.barh(bar_y, (0.60 - threshold) * 100, left=threshold * 100, height=bar_height, color='#d97706', alpha=0.35, label='Elevated Risk Zone')
            ax.barh(bar_y, (1.0 - 0.60) * 100, left=60, height=bar_height, color='#dc2626', alpha=0.35, label='High Risk Zone (Default)')

            # Calibrated Decision Threshold Marker
            ax.axvline(threshold * 100, color='#f59e0b', linestyle='--', linewidth=2.5,
                       label=f'Calibrated Decision Cutoff ({threshold*100:.0f}%)')

            # Current Applicant Risk Marker
            current_pct = prob_default * 100
            marker_color = '#ef4444' if prob_default >= threshold else '#10b981'
            ax.scatter([current_pct], [bar_y], color=marker_color, s=260, zorder=5, edgecolors='#ffffff', linewidths=2.5)

            # Arrow Annotation
            verdict_badge = "DEFAULT PREDICTED" if prob_default >= threshold else "SAFE (NON-DEFAULT)"
            ax.annotate(f"Applicant: {current_pct:.1f}%\n[{verdict_badge}]",
                        xy=(current_pct, bar_y + 0.18),
                        xytext=(current_pct, bar_y + 0.55),
                        ha='center',
                        color=marker_color,
                        fontsize=10.5,
                        fontweight='bold',
                        arrowprops=dict(facecolor=marker_color, shrink=0.08, width=2, headwidth=7),
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1e293b', edgecolor=marker_color, alpha=0.9))

            ax.set_xlim(0, 100)
            ax.set_ylim(0.1, 1.25)
            ax.set_yticks([])
            ax.tick_params(axis='x', colors='#94a3b8', labelsize=10)
            ax.set_xlabel('Default Risk Probability (%)', color='#94a3b8', fontsize=10, labelpad=8)
            ax.set_title('Proof 2: Default Risk Gauge vs Policy Decision Threshold',
                         color='#f8fafc', fontsize=12.5, fontweight='bold', pad=15)

            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')

            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=4,
                      frameon=False, fontsize=8.5, labelcolor='#94a3b8')

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error plotting risk gauge: {e}")
            return None

    def _plot_benchmark(self, raw_input_df, engineered_df):
        """
        Proof Plot 3: Compares Applicant against Historical Default vs Non-Default population averages.
        """
        try:
            # Extract applicant values
            applicant_limit = float(raw_input_df['LIMIT_BAL'].iloc[0])
            applicant_pay0 = float(raw_input_df['PAY_0'].iloc[0])
            applicant_avg_delay = float(engineered_df['AVG_PAY_DELAY'].iloc[0]) if 'AVG_PAY_DELAY' in engineered_df.columns else float(raw_input_df['PAY_0'].iloc[0])
            applicant_util = float(engineered_df['CREDIT_UTILIZATION'].iloc[0]) if 'CREDIT_UTILIZATION' in engineered_df.columns else (float(raw_input_df['BILL_AMT1'].iloc[0]) / (applicant_limit + 1.0))
            applicant_pay1 = float(raw_input_df['PAY_AMT1'].iloc[0])

            # Normalize values for fair 0-100 visual comparison
            # Metric 1: Credit Limit (0 to 300,000)
            m1_app = min(100, (applicant_limit / 300000.0) * 100)
            m1_non = min(100, (BASELINE_STATS['non_default']['LIMIT_BAL'] / 300000.0) * 100)
            m1_def = min(100, (BASELINE_STATS['default']['LIMIT_BAL'] / 300000.0) * 100)

            # Metric 2: Recent Payment Delay PAY_0 (-1 to 3 -> mapped 0 to 100)
            m2_app = np.clip(((applicant_pay0 + 1) / 4.0) * 100, 0, 100)
            m2_non = np.clip(((BASELINE_STATS['non_default']['PAY_0'] + 1) / 4.0) * 100, 0, 100)
            m2_def = np.clip(((BASELINE_STATS['default']['PAY_0'] + 1) / 4.0) * 100, 0, 100)

            # Metric 3: Credit Utilization % (0 to 1.0 -> 0 to 100)
            m3_app = np.clip(applicant_util * 100, 0, 100)
            m3_non = np.clip(BASELINE_STATS['non_default']['CREDIT_UTILIZATION'] * 100, 0, 100)
            m3_def = np.clip(BASELINE_STATS['default']['CREDIT_UTILIZATION'] * 100, 0, 100)

            # Metric 4: Recent Payment Amount (0 to 10,000)
            m4_app = min(100, (applicant_pay1 / 10000.0) * 100)
            m4_non = min(100, (BASELINE_STATS['non_default']['PAY_AMT1'] / 10000.0) * 100)
            m4_def = min(100, (BASELINE_STATS['default']['PAY_AMT1'] / 10000.0) * 100)

            categories = ['Credit Limit\n(Higher is Safer)', 'Payment Delay\n(Higher is Riskier)', 'Credit Utilization\n(Higher is Riskier)', 'Payment Amount\n(Higher is Safer)']
            applicant_vals = [m1_app, m2_app, m3_app, m4_app]
            non_def_vals = [m1_non, m2_non, m3_non, m4_non]
            def_vals = [m1_def, m2_def, m3_def, m4_def]

            fig, ax = plt.subplots(figsize=(10, 4.8), facecolor='#0f172a')
            ax.set_facecolor('#1e293b')

            x = np.arange(len(categories))
            width = 0.25

            r1 = ax.bar(x - width, applicant_vals, width, label='Current Applicant', color='#38bdf8', edgecolor='none')
            r2 = ax.bar(x, non_def_vals, width, label='Historical Non-Default Avg', color='#10b981', edgecolor='none')
            r3 = ax.bar(x + width, def_vals, width, label='Historical Default Avg', color='#ef4444', edgecolor='none')

            ax.set_ylabel('Normalized Score (0-100 Index)', color='#94a3b8', fontsize=9.5)
            ax.set_title('Proof 3: Comparative Benchmark vs Historical Population Groups',
                         color='#f8fafc', fontsize=12.5, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, color='#f8fafc', fontsize=9.5, fontweight='bold')
            ax.tick_params(axis='y', colors='#94a3b8')
            ax.set_ylim(0, 115)

            # Add raw values on top of applicant bars
            raw_labels = [f"NT${applicant_limit:,.0f}", f"{applicant_pay0:.0f} mo", f"{applicant_util*100:.0f}%", f"NT${applicant_pay1:,.0f}"]
            for bar, text in zip(r1, raw_labels):
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 3, text,
                        ha='center', va='bottom', color='#38bdf8', fontsize=8.5, fontweight='bold')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#334155')
            ax.spines['bottom'].set_color('#334155')

            ax.legend(loc='upper right', framealpha=0.4, facecolor='#0f172a', edgecolor='#334155',
                      labelcolor='#f8fafc', fontsize=9)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error plotting benchmark: {e}")
            return None

