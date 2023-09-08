import pyperclip as pc
import pandas as pd


def format_num(num: float) -> str:
    """helper function to format the string as four decimal places
    """
    num = round(num, 4)
    if len(str(num)) < 6:
        return "%s0" % num
    else:
        return str(num)


def is_float(string) -> bool:
    """helper function to determine if a string is in float format
    """
    try:
        float(string)
        return True
    except ValueError:
        return False


def table_gen(exp, df, label):
    """
    Function for generating the ranking performance table in latex
    :param exp: the experiment number
    :param df: the loaded csv file in dataframe
    :param label: a unique label for identification
    """

    def add_rows(metric, idx, tmp_rows, metric_p) -> str:
        """
        Inner function to avoid calling the same lines repeatedly
        :param metric: evaluation results from one of the four metrics: [nDCG@10, nDCG@100, AP, RR]
        :param idx: current index to access the value
        :param tmp_rows: temporary variable to store the expression of the rows
        :param metric_p: the p values of a metric
        :return: String - a new tmp_rows
        """
        value_metric = format_num(metric[idx])
        if metric_p[idx] < 0.01:
            tmp_rows += "$\downarrow$ "
        if metric[idx] == max(metric):
            tmp_rows += "$\\mathbf{%s}$ & " % value_metric
        else:
            tmp_rows += "%s & " % value_metric

        return tmp_rows

    # template for generating the latex table
    tmpl = """\\begin{table}[htbp]
\\centering
\\caption{Breakdown of ranking performance from Experiment """ + f'{exp}' + """. Statistically significant differences with the base model are indicated by $\downarrow$ (paired t-test by query, $p < 0.01$).}
\\label{tab:table-""" + f'{label}' + """}
\\vspace{5mm}
\\begin{tabular}{lrrrrr}
\\hline
\\\ [-1.5ex]
Ranker & nDCG@10 & nDCG@100 & AP(rel=2) & RR(rel=2) \\\\ [1ex]
\\hline [insert]
\\hline
\\end{tabular}
\\end{table}"""

    tmp_rows = ""
    cell_text = []
    for i, column in enumerate(df.columns):  # formatting the data
        tmp = [column]
        for n in df[column]:
            tmp.append(n)
        cell_text.append(tmp)

    for i, r in enumerate(cell_text[0][1:]):  # generating the rows line by line from 'Base' to 'Hard'
        r = r.replace('%', '')
        if r == "0":
            tmp_rows += "Base & "
        elif r == "100":
            tmp_rows += "Hard & "
        else:
            tmp_rows += f"$r$ = 0.{r[0]} & "

        ndcg_10 = cell_text[3][1:]
        ndcg_10_p = cell_text[13][1:]
        ndcg_100 = cell_text[4][1:]
        ndcg_100_p = cell_text[16][1:]
        ap = cell_text[1][1:]
        ap_p = cell_text[7][1:]
        rr = cell_text[2][1:]
        rr_p = cell_text[10][1:]

        tmp_rows = add_rows(ndcg_10, i, tmp_rows, ndcg_10_p)
        tmp_rows = add_rows(ndcg_100, i, tmp_rows, ndcg_100_p)
        tmp_rows = add_rows(ap, i, tmp_rows, ap_p)
        tmp_rows = add_rows(rr, i, tmp_rows, rr_p)
        tmp_rows = tmp_rows[:-2] + "\\\\\n"

        result = tmpl.replace('[insert]', tmp_rows)

        # copy the lines to the clipboard for pasting
        pc.copy(result)


if __name__ == '__main__':
    result_df = pd.read_csv(f'exp_4_test-2019_20230807_16.csv', index_col=0)  # evaluation result csv file
    table_gen(4, result_df, "experiment-4")
