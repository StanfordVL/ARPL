# this file provides helper functions to write a LaTex document with table results

def init_doc(title):
    print("\\documentclass[11pt]{article}")
    print("\\usepackage{amssymb}")
    print("\\usepackage{mathrsfs,amsmath}")
    print("\\usepackage{graphicx}")
    print("\\usepackage{amsmath}")
    print("\\usepackage{amsthm}")
    print("\\usepackage{bbm}")
    print("\\usepackage{dsfont}")
    print("\\usepackage{listing}")
    print("\\usepackage{array} %for table entries to be in center of cell")
    print("\\usepackage{tabularx}")
    print("\\usepackage{multirow}")
    print("\\usepackage{multicol}")
    print("\\usepackage{booktabs}")
    print("\\usepackage{color}")
    print("\\usepackage{colortbl}")
    print("\\usepackage{xcolor}")
    print("")
    print("\\extrafloats{100}")
    print("")
    print("\\usepackage{fancyhdr}")
    print("\\pagestyle{fancy}")
    print("\\usepackage{calc}")
    print("\\fancyheadoffset[RE]{\\marginparsep+\\marginparwidth}")
    print("\\usepackage[top=1.5in, bottom=1in, left=0.8in, right=0.8in]{geometry}")
    print("\\usepackage{lscape}")
    print("")
    print("\\title{" + title + "}")
    print("\\author{Ajay Mandlekar}")
    print("\\begin{document}")
    print("\\maketitle")

def end_doc():
    print("\\end{document}")

def print_table_normal(table, col_headers, row_headers, caption):
    """

    :param table: Numeric 2-d array to display as a table.
    :param col_headers: List of strings, titles of each column.
    :param row_headers: List of strings, titles of each row (except the first row).
    :param caption: A string to use for the caption.
    :return:
    """

    # map each row element to a string
    col_mapper = lambda x : '& {}'.format(round(x, 1))
    row_mapper = lambda x : ' '.join(list(map(col_mapper, x)))
    table_strs = list(map(row_mapper, table))

    assert(len(table_strs) == len(row_headers))

    print("\\begin{landscape}")
    print("\\begin{table}[h]")
    print("\\caption{" + caption + "}")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{" + "| c " * len(col_headers) + "|}")
    print("\\hline")
    print(' & '.join(col_headers) + "\\\\ \\hline")
    for i in range(len(row_headers)):
        print(row_headers[i] + " " + table_strs[i] + " \\\\ \\hline")
    print("\\end{tabular}}")
    print("\\end{table}")
    print("\\end{landscape}")


def print_table(rollouts, env_name, phi1, phi2, eps, num_iters, is_mean=True):
    assert(len(rollouts) == 13)
    assert(len(rollouts[0]) == 13)

    # map each row element to a string
    col_mapper = lambda x : '& {}'.format(round(x, 1))
    row_mapper = lambda x : ' '.join(list(map(col_mapper, x)))
    rollout_strs = list(map(row_mapper, rollouts))

    if is_mean:
        str1 = "mean of cumulative rewards"
    else:
        str1 = "standard deviation of cumulative rewards"

    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Performance (" + str1 + ") on \\texttt{" + env_name + "} under different types of perturbations, with $\epsilon = " + str(eps) + "$. Different training regimes are "
          "listed in rows and tested against different types of perturbations in columns."
          " Agents were trained with " + str(num_iters) + " iterations of TRPO.}")
    print("\\label{tab:main}")
    print("\\resizebox{\\linewidth}{!}{% put in textwidth")
    print("\\begin{tabular}{cccccccccccccccc} ")
    print("& & & & \\multicolumn{6}{c}{\\cellcolor[HTML]{CBCEFB}$\\phi$ = " + str(phi1) + "} & "
          "\\multicolumn{6}{c}{\\cellcolor[HTML]{CBCEFB}$\\phi$ = " + str(phi2) + "}      ")
    print("\\\\")
    print("& & & & \\multicolumn{2}{c}{\\cellcolor[HTML]{FFCB2F}Dynamics Noise} ")
    print("& \\multicolumn{2}{c}{\\cellcolor[HTML]{9AFF99}Process Noise} ")
    print("& \\multicolumn{2}{c}{\\cellcolor[HTML]{FFCCC9}Observation Noise} ")
    print("& \\multicolumn{2}{c}{\\cellcolor[HTML]{FFCB2F}Dynamics Noise} ")
    print("& \\multicolumn{2}{c}{\\cellcolor[HTML]{9AFF99}Process Noise} ")
    print("& \\multicolumn{2}{c}{\\cellcolor[HTML]{FFCCC9}Observation Noise} \\\\")
    print("& & & \\multirow{-3}{*}{Nominal} & Random & Adversarial & Random & Adversarial & Random ")
    print("& Adversarial & Random & Adversarial & Random & Adversarial & Random & Adversarial")
    print("\\\\")
    print("\\multicolumn{3}{c}{Nominal}     ")
    # row 1
    print(rollout_strs[0])

    print("\\\\")
    print("\\rowcolor[HTML]{EFEFEF} ")
    print("\\cellcolor[HTML]{CBCEFB} & \\cellcolor[HTML]{FFCB2F} & Random     ")

    # row 2
    print(rollout_strs[1])

    print("\\\\")
    print("\\cellcolor[HTML]{CBCEFB} & \\multirow{-2}{*}{\\cellcolor[HTML]{FFCB2F}Dynamics Noise} & Adversarial ")

    # row 3
    print(rollout_strs[2])

    print("\\\\")
    print("\\rowcolor[HTML]{EFEFEF} ")
    print("\\cellcolor[HTML]{CBCEFB} & \\cellcolor[HTML]{9AFF99} & Random      ")

    # row 4
    print(rollout_strs[3])

    print("\\\\")
    print("\\cellcolor[HTML]{CBCEFB} & \\multirow{-2}{*}{\\cellcolor[HTML]{9AFF99}Process Noise} & Adversarial ")

    # row 5
    print(rollout_strs[4])

    print("\\\\")
    print("\\rowcolor[HTML]{EFEFEF} ")
    print("\\cellcolor[HTML]{CBCEFB} & \\cellcolor[HTML]{FFCCC9} & Random ")

    # row 6
    print(rollout_strs[5])

    print("\\\\")
    print("\\multirow{-6}{*}{\\cellcolor[HTML]{CBCEFB}$\\phi$ = " + str(phi1) + "} & "
          "\\multirow{-2}{*}{\\cellcolor[HTML]{FFCCC9}Observation Noise} & Adversarial ")

    # row 7
    print(rollout_strs[6])

    print("\\\\")
    print("\\rowcolor[HTML]{EFEFEF} ")
    print("\\cellcolor[HTML]{CBCEFB} & \\cellcolor[HTML]{FFCB2F} & Random      ")

    # row 8
    print(rollout_strs[7])

    print("\\\\")
    print("\\cellcolor[HTML]{CBCEFB} & \\multirow{-2}{*}{\\cellcolor[HTML]{FFCB2F}Dynamics Noise} & Adversarial ")

    # row 9
    print(rollout_strs[8])

    print("\\\\")
    print("\\rowcolor[HTML]{EFEFEF} ")
    print("\\cellcolor[HTML]{CBCEFB} & \\cellcolor[HTML]{9AFF99} & Random    ")

    # row 10
    print(rollout_strs[9])

    print("\\\\")
    print("\\cellcolor[HTML]{CBCEFB} & \\multirow{-2}{*}{\\cellcolor[HTML]{9AFF99}Process Noise} & Adversarial")

    # row 11
    print(rollout_strs[10])

    print("\\\\")
    print("\\rowcolor[HTML]{EFEFEF} ")
    print("\\cellcolor[HTML]{CBCEFB} & \\cellcolor[HTML]{FFCCC9} & Random  ")

    # row 12
    print(rollout_strs[11])

    print("\\\\")
    print("\\multirow{-6}{*}{\\cellcolor[HTML]{CBCEFB}$\\phi$ = " + str(phi2) + "} & "
          "\\multirow{-2}{*}{\\cellcolor[HTML]{FFCCC9}Observation Noise} & Adversarial ")

    # row 13
    print(rollout_strs[12])

    print("\\end{tabular}")
    print("}")
    print("\\vspace{-10pt}")
    print("\\end{table*}")

