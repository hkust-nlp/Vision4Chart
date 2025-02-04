FIGURE_QA_PROMPT="{Question}\nAnswer the question with yes or no."
CHARTBENCH_NQA_PROMPT="{Question}\nAnswer the question using one value."
CHARTBENCH_BIN_PROMPT="You need to determine whether the following is correct.\n{Question}\nIs it correct? Answer with yes or no."
DVQA_PROMPT="{Question}\nAnswer the question with a single word."
PLOT_PROMPT="{Question}\nAnswer the question with a single word."
MMC_BENCH_PROMPT="{Question}\nAnswer this question using a single word: true or false."
CHARTQA_PROMPT="{Question}\nAnswer the question with a single word."
CHARTX_PROMPT="{Question}\nAnswer the question using a single word or phrase."
prompt_templates = {
    "fqa":FIGURE_QA_PROMPT,
    "chartbench_nqa":CHARTBENCH_NQA_PROMPT,
    "chartbench_bin":CHARTBENCH_BIN_PROMPT,
    "dvqa":DVQA_PROMPT,
    "plotqa":PLOT_PROMPT,
    "mmc_bench":MMC_BENCH_PROMPT,
    "chartqa":CHARTQA_PROMPT,
    "chartX":CHARTX_PROMPT
}

if __name__ == '__main__':
    pass