
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# إعداد الصفحة
st.set_page_config(
    page_title="Research Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Research Analyzer by Amatalelah")
st.subheader("Statistical Analysis Platform | منصة التحليل الإحصائي")
st.write("Welcome, Researchers and Students!")
st.divider()


# دالة عرض النتائج
def show_result(name, result):
    st.subheader(name)
    st.write("Test statistic:", round(float(result.statistic), 5))
    st.write("P-value:", round(float(result.pvalue), 5))

    if result.pvalue < 0.05:
        st.success("Statistically significant (p < 0.05)")
    else:
        st.info("Not statistically significant (p ≥ 0.05)")


# رفع البيانات
file = st.file_uploader(
    "Upload CSV or Excel file | ارفعي ملف البيانات",
    type=["csv", "xlsx"]
)

if file is not None:

    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        if df.empty:
            st.error("The uploaded file is empty.")
            st.stop()

        st.success("Data uploaded successfully!")

        # 1. نظرة عامة
        st.header("1. Data Overview | نظرة عامة")

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric(
            "Missing Values",
            int(df.isna().sum().sum())
        )

        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True)

        st.divider()

        # 2. جودة البيانات
        st.header("2. Data Quality | جودة البيانات")

        quality = pd.DataFrame({
            "Variable": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isna().sum().values,
            "Missing %": (
                df.isna().mean() * 100
            ).round(2).values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(quality, use_container_width=True)

        st.download_button(
            "Download Data Quality Report",
            quality.to_csv(index=False).encode("utf-8-sig"),
            "data_quality.csv",
            "text/csv"
        )

        st.divider()

        # 3. الإحصاء الوصفي
        st.header("3. Descriptive Statistics")

        numeric = df.select_dtypes(include=np.number)
        numeric_cols = numeric.columns.tolist()

        if not numeric.empty:
            descriptive = numeric.describe().T.round(3)
            st.dataframe(
                descriptive,
                use_container_width=True
            )
        else:
            st.info("No numeric variables found.")

        st.divider()

        # 4. الارتباط
        st.header("4. Correlation Matrix")

        if len(numeric_cols) >= 2:

            method = st.selectbox(
                "Correlation method",
                ["pearson", "spearman", "kendall"]
            )

            corr = numeric.corr(method=method).round(3)
            st.dataframe(corr, use_container_width=True)

        else:
            st.info("At least two numeric variables are required.")

        st.divider()

        # 5. الاختبارات الإحصائية
        st.header("5. Statistical Tests | الاختبارات الإحصائية")

        tests = [
            "One-Sample t-test",
            "Independent t-test",
            "Paired t-test",
            "One-Way ANOVA",
            "Mann-Whitney U",
            "Wilcoxon Signed-Rank",
            "Kruskal-Wallis",
            "Friedman Test",
            "Chi-Square",
            "Fisher Exact",
            "McNemar Test",
            "Shapiro-Wilk Normality",
            "Levene Test"
        ]

        test = st.selectbox(
            "Choose Statistical Test",
            tests
        )

        group_cols = [
            c for c in df.columns
            if 2 <= df[c].nunique() <= 10
        ]

        # اختبار t لعينة واحدة
        if test == "One-Sample t-test":

            if numeric_cols:

                col = st.selectbox(
                    "Select numeric variable",
                    numeric_cols
                )

                mean = st.number_input(
                    "Hypothesized mean",
                    value=0.0
                )

                if st.button("Run Test"):

                    x = df[col].dropna()

                    if len(x) >= 2:
                        show_result(
                            test,
                            stats.ttest_1samp(
                                x,
                                popmean=mean
                            )
                        )
                    else:
                        st.warning(
                            "At least two observations are required."
                        )

        # اختبار t لمجموعتين مستقلتين
        elif test == "Independent t-test":

            if numeric_cols and group_cols:

                value = st.selectbox(
                    "Numeric variable",
                    numeric_cols
                )

                group = st.selectbox(
                    "Grouping variable",
                    group_cols
                )

                if st.button("Run Test"):

                    data = df[[value, group]].dropna()
                    groups = data[group].unique()

                    if len(groups) == 2:

                        a = data[
                            data[group] == groups[0]
                        ][value]

                        b = data[
                            data[group] == groups[1]
                        ][value]

                        if len(a) >= 2 and len(b) >= 2:

                            result = stats.ttest_ind(
                                a, b,
                                equal_var=False
                            )

                            show_result(
                                "Welch Independent t-test",
                                result
                            )

                        else:
                            st.warning(
                                "Each group needs at least two observations."
                            )

                    else:
                        st.warning(
                            "Exactly two groups are required."
                        )

        # اختبار t للعينات المزدوجة
        elif test == "Paired t-test":

            if len(numeric_cols) >= 2:

                a_col = st.selectbox(
                    "First measurement",
                    numeric_cols
                )

                b_col = st.selectbox(
                    "Second measurement",
                    [c for c in numeric_cols if c != a_col]
                )

                if st.button("Run Test"):

                    data = df[[a_col, b_col]].dropna()

                    if len(data) >= 2:

                        show_result(
                            test,
                            stats.ttest_rel(
                                data[a_col],
                                data[b_col]
                            )
                        )

                    else:
                        st.warning(
                            "At least two complete pairs are required."
                        )

        # ANOVA و Kruskal-Wallis
        elif test in [
            "One-Way ANOVA",
            "Kruskal-Wallis"
        ]:

            if numeric_cols and group_cols:

                value = st.selectbox(
                    "Numeric variable",
                    numeric_cols
                )

                group = st.selectbox(
                    "Grouping variable",
                    group_cols
                )

                if st.button("Run Test"):

                    data = df[[value, group]].dropna()

                    samples = [
                        part[value].values
                        for _, part in data.groupby(group)
                    ]

                    if (
                        len(samples) >= 2
                        and all(len(s) >= 2 for s in samples)
                    ):

                        if test == "One-Way ANOVA":
                            result = stats.f_oneway(*samples)
                        else:
                            result = stats.kruskal(*samples)

                        show_result(test, result)

                    else:
                        st.warning(
                            "At least two groups with two observations each are required."
                        )

        # Mann-Whitney
        elif test == "Mann-Whitney U":

            if numeric_cols and group_cols:

                value = st.selectbox(
                    "Numeric variable",
                    numeric_cols
                )

                group = st.selectbox(
                    "Grouping variable",
                    group_cols
                )

                if st.button("Run Test"):

                    data = df[[value, group]].dropna()
                    groups = data[group].unique()

                    if len(groups) == 2:

                        a = data[
                            data[group] == groups[0]
                        ][value]

                        b = data[
                            data[group] == groups[1]
                        ][value]

                        if len(a) > 0 and len(b) > 0:

                            show_result(
                                test,
                                stats.mannwhitneyu(
                                    a, b,
                                    alternative="two-sided"
                                )
                            )

                        else:
                            st.warning(
                                "Both groups need observations."
                            )

                    else:
                        st.warning(
                            "Exactly two groups are required."
                        )

        # Wilcoxon
        elif test == "Wilcoxon Signed-Rank":

            if len(numeric_cols) >= 2:

                a_col = st.selectbox(
                    "First measurement",
                    numeric_cols
                )

                b_col = st.selectbox(
                    "Second measurement",
                    [c for c in numeric_cols if c != a_col]
                )

                if st.button("Run Test"):

                    data = df[[a_col, b_col]].dropna()

                    if len(data) >= 2:

                        try:
                            show_result(
                                test,
                                stats.wilcoxon(
                                    data[a_col],
                                    data[b_col]
                                )
                            )

                        except ValueError as e:
                            st.warning(str(e))

                    else:
                        st.warning(
                            "At least two complete pairs are required."
                        )

        # Friedman
        elif test == "Friedman Test":

            if len(numeric_cols) >= 3:

                cols = st.multiselect(
                    "Select three or more paired measurements",
                    numeric_cols
                )

                if st.button("Run Test"):

                    if len(cols) >= 3:

                        data = df[cols].dropna()

                        if len(data) >= 2:

                            show_result(
                                test,
                                stats.friedmanchisquare(
                                    *[
                                        data[c]
                                        for c in cols
                                    ]
                                )
                            )

                        else:
                            st.warning(
                                "At least two complete cases are required."
                            )

                    else:
                        st.warning(
                            "Select at least three measurements."
                        )

        # Chi-square و Fisher و McNemar
        elif test in [
            "Chi-Square",
            "Fisher Exact",
            "McNemar Test"
        ]:

            if len(df.columns) >= 2:

                a_col = st.selectbox(
                    "First categorical variable",
                    df.columns
                )

                b_col = st.selectbox(
                    "Second categorical variable",
                    [
                        c for c in df.columns
                        if c != a_col
                    ]
                )

                if st.button("Run Test"):

                    table = pd.crosstab(
                        df[a_col],
                        df[b_col]
                    )

                    if test == "Chi-Square":

                        if (
                            table.shape[0] >= 2
                            and table.shape[1] >= 2
                        ):

                            result = stats.chi2_contingency(
                                table
                            )

                            st.write(
                                "Chi-square:",
                                round(result.statistic, 5)
                            )

                            st.write(
                                "P-value:",
                                round(result.pvalue, 5)
                            )

                            st.write(
                                "Degrees of freedom:",
                                result.dof
                            )

                            st.write("Expected frequencies:")

                            st.dataframe(
                                pd.DataFrame(
                                    result.expected_freq,
                                    index=table.index,
                                    columns=table.columns
                                )
                            )

                        else:
                            st.warning(
                                "At least two categories per variable are required."
                            )

                    elif test == "Fisher Exact":

                        if table.shape == (2, 2):

                            result = stats.fisher_exact(table)
                            show_result(test, result)

                        else:
                            st.warning(
                                "Fisher Exact requires a 2 × 2 table."
                            )

                    elif test == "McNemar Test":

                        if table.shape == (2, 2):

                            result = mcnemar(
                                table,
                                exact=True
                            )

                            show_result(test, result)

                            st.caption(
                                "Use only for paired categorical measurements "
                                "from the same participants."
                            )

                        else:
                            st.warning(
                                "McNemar requires paired binary data producing a 2 × 2 table."
                            )

        # اختبار Shapiro-Wilk
        elif test == "Shapiro-Wilk Normality":

            if numeric_cols:

                col = st.selectbox(
                    "Select numeric variable",
                    numeric_cols
                )

                if st.button("Run Test"):

                    x = df[col].dropna()

                    if 3 <= len(x) <= 5000:

                        show_result(
                            test,
                            stats.shapiro(x)
                        )

                    elif len(x) > 5000:
                        st.warning(
                            "This version supports up to 5000 observations for Shapiro-Wilk."
                        )

                    else:
                        st.warning(
                            "At least three observations are required."
                        )

        # اختبار Levene
        elif test == "Levene Test":

            if numeric_cols and group_cols:

                value = st.selectbox(
                    "Numeric variable",
                    numeric_cols
                )

                group = st.selectbox(
                    "Grouping variable",
                    group_cols
                )

                if st.button("Run Test"):

                    data = df[[value, group]].dropna()

                    samples = [
                        part[value].values
                        for _, part in data.groupby(group)
                    ]

                    if len(samples) >= 2 and all(
                        len(s) >= 2 for s in samples
                    ):

                        show_result(
                            test,
                            stats.levene(*samples)
                        )

                    else:
                        st.warning(
                            "At least two groups with two observations each are required."
                        )

        else:
            st.info(
                "Select the required variables to perform the test."
            )

        st.divider()

        # 6. تحميل النتائج
        st.header("6. Download Results")

        st.download_button(
            "Download Data Quality Report",
            quality.to_csv(index=False).encode("utf-8-sig"),
            "data_quality.csv",
            "text/csv"
        )

        st.download_button(
            "Download Descriptive Statistics",
            numeric.describe().T.to_csv().encode("utf-8-sig")
            if not numeric.empty else "",
            "descriptive_statistics.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info(
        "Upload a CSV or Excel file to begin."
    )
