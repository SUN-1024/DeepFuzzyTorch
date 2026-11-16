"""
Transforms the finalized primary manuscript (`paper/manuscript_Information_Fusion.tex`)
into two structurally compliant, mutually exclusive single-submission venue templates:
  1. `paper/manuscript_Genome_Biology_Software.tex` (BMC Genome Biology `Software` format)
  2. `paper/manuscript_Nature_Communications.tex` (Nature Portfolio `Results-first` format)

Note: The three `paper/manuscript_*.tex` files represent alternative single-venue formatting
templates of the same study (Information Fusion primary) and must never be co-submitted.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INF_FUS_TEX = os.path.join(WORKSPACE_DIR, "paper/manuscript_Information_Fusion.tex")
DEFAULT_OUT_DIR = os.path.join(WORKSPACE_DIR, "paper")


def _find_section_pos(text: str, candidates: List[str], start: int = 0) -> int:
    best = -1
    for cand in candidates:
        idx = text.find(cand, start)
        if idx != -1 and (best == -1 or idx < best):
            best = idx
    if best == -1:
        raise ValueError(f"Could not find any of section markers {candidates} starting at {start}.")
    return best


def _strip_trailing_floatbarrier(block: str) -> str:
    return re.sub(r"(?:\s*\\FloatBarrier\s*)+\Z", "\n", block.rstrip()) + "\n"


def update_hypersetup(preamble: str, pdftitle: str, pdfauthor: str = "Guanqun Sun") -> str:
    updated = re.sub(
        r"pdftitle=\{[^}]*\}",
        lambda _: f"pdftitle={{{pdftitle}}}",
        preamble,
        count=1,
    )
    updated = re.sub(
        r"pdfauthor=\{[^}]*\}",
        lambda _: f"pdfauthor={{{pdfauthor}}}",
        updated,
        count=1,
    )
    if f"pdfauthor={{{pdfauthor}}}" not in updated or "et al." in updated:
        raise AssertionError("Failed to sanitize pdfauthor metadata in \\hypersetup.")
    return updated


def parse_if_sections(src: str) -> Dict[str, str]:
    idx_docclass = _find_section_pos(src, [r"\documentclass"])
    idx_pagestyle = _find_section_pos(src, [r"\pagestyle{fancy}"], idx_docclass)
    idx_intro = _find_section_pos(src, [r"\section{Introduction}"], idx_pagestyle)
    idx_related = _find_section_pos(src, [r"\section{Related Work}"], idx_intro)
    idx_method = _find_section_pos(src, [r"\section{Methodology}"], idx_related)
    idx_exp = _find_section_pos(src, [r"\section{Experimental Setup}"], idx_method)
    idx_results = _find_section_pos(
        src, [r"\section{Experimental Results}", r"\section{Results}"], idx_exp
    )
    idx_disc = _find_section_pos(src, [r"\section{Discussion}"], idx_results)
    idx_conc = _find_section_pos(
        src, [r"\section{Conclusion}", r"\section{Conclusions}"], idx_disc
    )
    idx_app = _find_section_pos(src, [r"\appendix"], idx_conc)
    idx_end = _find_section_pos(src, [r"\end{document}"], idx_app)

    conc_block = _strip_trailing_floatbarrier(src[idx_conc:idx_app])
    # Strip any optional unnumbered backmatter (\section*{...}) after Conclusion so we
    # never duplicate or misplace availability/competing-interest blocks.
    opt_backmatter = re.search(r"\n\\section\*\{", conc_block)
    if opt_backmatter:
        conc_core = conc_block[: opt_backmatter.start()].strip()
    else:
        conc_core = conc_block.strip()

    # Remove the \section{Conclusion(s)} header and optional \label{sec:conclusion}
    conc_body_only = re.sub(
        r"^\\section\{Conclusions?\}\s*(?:\\label\{[^}]+\}\s*)?",
        "",
        conc_core,
        count=1,
    ).strip()

    # Separate scientific concluding synthesis from artifact-bundling / URL sentences
    avail_split = re.search(
        r"(?:\n\s*\n|\s+)(?=(?:Complete source code|All source code|Source code, pre-extracted|Code and data))",
        conc_body_only,
    )
    if avail_split:
        conc_synthesis = conc_body_only[: avail_split.start()].strip()
        conc_artifacts = conc_body_only[avail_split.start() :].strip()
    else:
        conc_synthesis = conc_body_only
        conc_artifacts = ""

    # Normalize any stale "planned for public release" wording if present
    conc_artifacts = conc_artifacts.replace(
        "and planned for public release at",
        "and publicly archived under the MIT License at",
    )

    return {
        "preamble": src[idx_docclass:idx_pagestyle],
        "intro": _strip_trailing_floatbarrier(src[idx_intro:idx_related]),
        "related": _strip_trailing_floatbarrier(src[idx_related:idx_method]),
        "method": _strip_trailing_floatbarrier(src[idx_method:idx_exp]),
        "experiments": _strip_trailing_floatbarrier(src[idx_exp:idx_results]),
        "results": _strip_trailing_floatbarrier(src[idx_results:idx_disc]),
        "discussion": _strip_trailing_floatbarrier(src[idx_disc:idx_conc]),
        "conc_synthesis": conc_synthesis,
        "conc_artifacts": conc_artifacts,
        "appendix": src[idx_app:idx_end].rstrip() + "\n",
    }


def extract_tables(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Extracts all table/table* environments from `text`, returning (remaining_text, [(label, table_tex)])."""
    tables: List[Tuple[str, str]] = []
    pattern = re.compile(r"\\begin\{(table\*?)\}.*?\\end\{\1\}\s*", flags=re.S)

    def _repl(match: re.Match[str]) -> str:
        tbl = match.group(0).strip()
        lbl_m = re.search(r"\\label\{([^}]+)\}", tbl)
        lbl = lbl_m.group(1) if lbl_m else ""
        tables.append((lbl, tbl))
        return "\n"

    cleaned = pattern.sub(_repl, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, tables


def extract_algorithms(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Extracts all algorithm/algorithm* environments from `text`, returning (remaining_text, [(label, alg_tex)])."""
    algs: List[Tuple[str, str]] = []
    pattern = re.compile(r"\\begin\{(algorithm\*?)\}.*?\\end\{\1\}\s*", flags=re.S)

    def _repl(match: re.Match[str]) -> str:
        alg = match.group(0).strip()
        lbl_m = re.search(r"\\label\{([^}]+)\}", alg)
        lbl = lbl_m.group(1) if lbl_m else ""
        algs.append((lbl, alg))
        return "\n"

    cleaned = pattern.sub(_repl, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned, algs


def partition_tables_for_nc(
    text: str, keep_labels: set[str], max_keep: int
) -> Tuple[str, List[Tuple[str, str]], List[str]]:
    """Keeps up to `max_keep` primary tables in-place and extracts secondary tables for Supplementary Information."""
    moved: List[Tuple[str, str]] = []
    kept_labels: List[str] = []
    pattern = re.compile(r"\\begin\{(table\*?)\}.*?\\end\{\1\}\s*", flags=re.S)

    def _repl(match: re.Match[str]) -> str:
        tbl = match.group(0).strip()
        lbl_m = re.search(r"\\label\{([^}]+)\}", tbl)
        lbl = lbl_m.group(1) if lbl_m else ""
        if lbl in keep_labels and len(kept_labels) < max_keep:
            kept_labels.append(lbl)
            return "\n" + tbl + "\n\n"
        moved.append((lbl, tbl))
        return "\n"

    updated = pattern.sub(_repl, text)
    updated = re.sub(r"\n{3,}", "\n\n", updated)
    return updated, moved, kept_labels


def _replace_ws_insensitive(text: str, old: str, new: str) -> str:
    """Replaces the first occurrence of ``old`` matching single spaces across arbitrary whitespace/newlines."""
    parts = [re.escape(tok) for tok in old.split()]
    if not parts:
        return text
    pattern = r"\s+".join(parts)
    return re.sub(pattern, lambda _: new, text, count=1)


def strip_cs_roadmap(intro_tex: str) -> str:
    """Removes the CS-style paper-organization roadmap paragraph from the end of Introduction."""
    return re.sub(
        r"\n\s*The remainder of this (?:paper|article|manuscript) is organized as follows\..*?(?=\n\s*\Z)",
        "\n",
        intro_tex,
        flags=re.S,
    )


def count_abstract_words(abs_tex: str) -> Tuple[int, int]:
    raw_wc = len(abs_tex.split())
    plain = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", abs_tex)
    plain = re.sub(r"\\[a-zA-Z]+\*?", " ", plain)
    plain = re.sub(r"[{}$~]", " ", plain)
    plain_wc = len(plain.split())
    return raw_wc, plain_wc


def assert_no_empty_same_level_headings(tex: str, name: str) -> None:
    """Verifies there are zero empty orphan headings (e.g., \\subsection immediately followed by \\subsection)."""
    no_comments = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in tex.splitlines())
    heading_re = re.compile(r"\\(section|subsection|subsubsection)\*?\{([^}]+)\}")
    levels = {"section": 1, "subsection": 2, "subsubsection": 3}
    matches = list(heading_re.finditer(no_comments))
    for i in range(len(matches) - 1):
        cur_m, nxt_m = matches[i], matches[i + 1]
        cur_lvl = levels[cur_m.group(1)]
        nxt_lvl = levels[nxt_m.group(1)]
        between = no_comments[cur_m.end() : nxt_m.start()]
        between_stripped = re.sub(r"\\label\{[^}]+\}", "", between)
        between_stripped = re.sub(r"\\FloatBarrier", "", between_stripped).strip()
        if nxt_lvl <= cur_lvl and not between_stripped:
            raise AssertionError(
                f"[{name}] Empty orphan heading detected: \\{cur_m.group(1)}{{{cur_m.group(2)}}} "
                f"immediately followed by \\{nxt_m.group(1)}{{{nxt_m.group(2)}}}!"
            )


def build_genome_biology_tex(parts: Dict[str, str]) -> str:
    gb_title_plain = (
        "DeepFuzzyTorch: An Open-Source Differentiable Fuzzy-Rough PyTorch Library "
        "for High-Dimensional Multi-Omics Reduction and Conflict-Aware Clinical Fusion"
    )
    gb_preamble = update_hypersetup(parts["preamble"], pdftitle=gb_title_plain, pdfauthor="Guanqun Sun")

    gb_header = (
        gb_preamble
        + r"""\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize \textit{Formatted for Genome Biology (Software)}}
\fancyhead[R]{\footnotesize \textit{DeepFuzzyTorch: Differentiable Fuzzy-Rough Multi-Omics \& Clinical Fusion}}
\fancyfoot[C]{\footnotesize \thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{\vspace{-1.0em}\textbf{DeepFuzzyTorch: An Open-Source Differentiable Fuzzy-Rough PyTorch Library for High-Dimensional Multi-Omics Reduction and Conflict-Aware Clinical Fusion}}

\author{
\textbf{Guanqun Sun}\\
\small School of Medical Imaging and Artificial Intelligence, Hangzhou Medical College, Hangzhou 310053, China\\
\small Open-Source Software Package \& Reproducibility Suite (MIT License): \url{https://github.com/SUN-1024/DeepFuzzyTorch}
}

\date{}

\begin{document}

\maketitle

\begin{abstract}
\subsubsection*{Background}
Multi-omics and multimodal biomedical integration---spanning high-dimensional pan-cancer transcriptomics (RNA-Seq), genomic instability signatures, gigapixel histopathology whole-slide images (WSIs), neuroimaging, and blood proteomics---is hindered by four coupled obstacles: combinatorial missing modalities across $2^M-1$ nonempty subsets at inference, real-world cross-modal biomarker discordance and physiological confounding, severe collinearity in high-dimensional transcriptomic and morphometric spaces ($D \ge 320\text{--}1,000$ features), and black-box fusion heads lacking auditable rule representations.

\subsubsection*{Results}
We present \texttt{DeepFuzzyTorch} (\texttt{v0.2.1}), an open-source PyTorch software library providing five composable PyTorch fuzzy operator blocks and an offline tensor-accelerated selector (\texttt{IFS\allowbreak Choquet\allowbreak Fusion}, \texttt{Hesitant\allowbreak Fuzzy\allowbreak Aggregator}, \texttt{Differentiable\allowbreak Fuzzy\allowbreak Rough\allowbreak Layer}/\allowbreak\texttt{Fuzzy\allowbreak Rough\allowbreak Feature\allowbreak Reducer}, \texttt{Deep\allowbreak Concept\allowbreak TSKHead}, and \texttt{Pythagorean\allowbreak Fuzzy\allowbreak TTALoss}) backed by five proved mathematical theorems and an end-to-end missing-modality/concept-mask invariance corollary ($0.00$ fixed-width padding-value discrepancy; algebraic submodel equivalence within float32 reduction rounding). Evaluated across 5,356 unique oncology and clinical participants and 4,849 histopathology patch/tile records against fourteen multimodal and tabular comparators (including \texttt{QMF (adapted)}, adapted \texttt{RCML}/\texttt{DrFuse}/\texttt{DDEF}/\texttt{FUML}, \texttt{XGBoost}, \texttt{CatBoost}, and \texttt{TabPFN~v2}), \texttt{FuzzyRoughFeatureReducer} performs supervised in-fold reduction of an unsupervised transductive 1,000-gene TCGA PanCancer Atlas MAD RNA-Seq pool to $K^*=20$ genes ($98.0\%$ compression, $94.05\%$ RNA branch parameter reduction, 5-fold pairwise Jaccard $0.3816$). This 20-gene reduct recovers 8 enriched Reactome pathways at Benjamini--Hochberg $\text{FDR}<0.05$ across all 676 predeclared $K_{\mathrm{bg}}\ge 2$ pathways in the custom 1,000-gene background ($q_{\mathrm{BH}}=0.0308$, led by \texttt{R-HSA-5686938} endogenous TLR regulation at $25.0\times$ enrichment) and improves 4-modality pan-cancer fusion over matched unreduced 1,000-gene fusion ($0.7403$ vs.\ $0.6925$ AUROC, $q_{\mathrm{BH}}=0.0025$; OS C-index $0.7514$ vs.\ $0.6933$). Across $N=1,112$ deduplicated unique ADNI participants, observed-pair conflict $C_{\mathcal{O}}$ is $1.51\times$ higher in $n=198$ real-world discordant patients than in $n=914$ concordant patients ($p = 1.75 \times 10^{-17}$), pure $C_{\mathcal{O}}$ detects $\gamma=0.8$ swaps at $0.7000$ AUROC ($0.6542$ composite), and nested matched-coverage triage ($81.3\%$, $161/198$) improves accuracy from $0.7475$ to $0.7764$ ($0.7826$ via pure $C_{\mathcal{O}}$; $0.7911$ at $80\%$). On ADNI Cohort~1, \texttt{DeepFuzzyTorch} reports separated natural-missingness ($N=1,137$, $0.8270$), same-visit ($n=924$, $0.8387$), and complete four-modality ($N_{\mathrm{comp}}=854$, $0.8371$) metrics alongside superior tree/foundation-model missingness retention ($\text{MRR}=0.9344$, 15-subset macro $0.7764$), 60-month dementia conversion ($0.9419$ AUROC; C-index $0.8804$), 5-seed external OASIS-3 transfer ($0.8639\pm 0.0010$ AUROC), corrupted-expert pathology consensus ($0.9925$ and $0.9804$ AUROC), and $2.384\times 10^{-7}$ text-rule reconstruction error.

\subsubsection*{Conclusions}
\texttt{DeepFuzzyTorch} delivers an installable, unit-tested PyTorch operator toolkit for granular multi-omics reduction, missing-modality-invariant fusion, conflict-guided clinical triage, and machine-precision white-box rule export.

\vspace{0.35em}
\noindent\textbf{Keywords:} Open-Source PyTorch Software; Multi-Omics Data Integration; Fuzzy-Rough Feature Selection; Custom-Background Pathway Enrichment; Missing Modalities; Biomarker Discordance; Computational Pathology.
\end{abstract}
"""
    )

    gb_roadmap = (
        "\nThe remainder of this article is organized as follows. "
        r"Subsection~\ref{sec:related} contrasts \texttt{DeepFuzzyTorch} with existing incomplete-modality, "
        r"evidential, and fuzzy-rough paradigms. Section~\ref{sec:method} details the software architecture, "
        r"five differentiable PyTorch operator blocks, and formal mathematical guarantees. "
        r"Section~\ref{sec:results} presents multi-center validation across neuroimaging, pan-cancer "
        r"transcriptomics, and computational pathology, followed by architectural synthesis and scope "
        r"limitations in Section~\ref{sec:discussion} and concluding remarks in Section~\ref{sec:conclusion}. "
        r"Section~\ref{sec:experiments} documents cohort provenance, zero-leakage preprocessing, and "
        r"statistical protocols, followed by mandatory software availability and editorial declarations in "
        r"Section~\ref{sec:declarations}, complete theorem proofs in Appendix~\ref{app:proofs}, and cohort "
        r"timestamp accounting in Appendix~\ref{app:provenance}." + "\n"
    )

    intro_clean = strip_cs_roadmap(parts["intro"]).replace(
        r"\section{Introduction}", r"\section{Background}", 1
    )
    # Demote \section{Related Work} -> \subsection{...} (1.1) AND all child \subsection -> \subsubsection (1.1.x)
    related_demoted = parts["related"].replace(r"\subsection{", r"\subsubsection{")
    related_demoted = related_demoted.replace(
        r"\section{Related Work}",
        r"\subsection{Comparison with Existing Multimodal and Evidential Paradigms}",
        1,
    )
    gb_background = intro_clean.rstrip() + "\n" + gb_roadmap + "\n" + related_demoted

    gb_implementation = parts["method"].replace(
        r"\section{Methodology}",
        r"\section{Implementation}",
        1,
    )
    gb_results = re.sub(
        r"^\\section\{(?:Experimental )?Results\}",
        r"\\section{Results}",
        parts["results"],
        count=1,
    )
    gb_discussion = parts["discussion"]
    gb_conclusions = (
        "\\section{Conclusions}\n\\label{sec:conclusion}\n"
        + parts["conc_synthesis"]
        + "\n"
    )
    gb_methods = parts["experiments"].replace(
        r"\section{Experimental Setup}",
        r"\section{Methods}",
        1,
    )

    artifact_note = (
        " " + parts["conc_artifacts"]
        if parts["conc_artifacts"]
        else ""
    )

    gb_declarations = (
        r"""\section{Declarations}
\label{sec:declarations}
The mandatory editorial, ethical, and software availability disclosures for \texttt{DeepFuzzyTorch} are detailed below.

\subsection*{Ethics approval and consent to participate}
All empirical benchmarks were conducted exclusively on de-identified, publicly available retrospective human neuroimaging, multi-omics, and histopathology cohorts (ADNI, OASIS-3, TCGA PanCancer Atlas, Camelyon17, and NCT-CRC). Institutional review board approval and written informed consent were obtained by the original study investigators at each participating clinical center; no new human participants or animal experiments were recruited for this study.

\subsection*{Consent for publication}
Not applicable.

\subsection*{Availability of data and materials}
\begin{itemize}[leftmargin=1.3em,itemsep=1pt,topsep=2pt]
    \item \textbf{Project name:} \texttt{DeepFuzzyTorch}
    \item \textbf{Project home page:} \url{https://github.com/SUN-1024/DeepFuzzyTorch}
    \item \textbf{Operating system(s):} Platform independent (Linux, macOS, Windows)
    \item \textbf{Programming language:} Python $\ge 3.9$ with PyTorch $\ge 2.0$ (CPU, CUDA GPU, and Apple Metal MPS supported)
    \item \textbf{Other requirements:} \texttt{numpy}, \texttt{scipy}, \texttt{scikit-learn}, \texttt{pandas}, \texttt{matplotlib}; optional benchmark comparators: \texttt{xgboost>=1.7.0}, \texttt{catboost>=1.2.0}, and \texttt{tabpfn>=2.0.0}
    \item \textbf{License:} MIT License
    \item \textbf{Any restrictions to use by non-academics:} None
\end{itemize}
Primary public datasets are accessible through the Alzheimer's Disease Neuroimaging Initiative (\url{https://adni.loni.usc.edu/}), OASIS-3 (\url{https://www.oasis-brains.org/}), the NCI Genomic Data Commons and cBioPortal TCGA PanCancer Atlas (\url{https://www.cbioportal.org/}), the Reactome Knowledgebase (\url{https://reactome.org/}), Camelyon17, and Zenodo NCT-CRC-HE-100K / CRC-VAL-HE-7K. Participant-level ADNI and OASIS-3 tables cannot be redistributed directly under their Data Use Agreements. Local benchmark scripts consume harmonized credentialed CSV exports and pre-extracted feature caches rather than bundling raw LONI neuroimaging or raw WSI ingestion pipelines."""
        + artifact_note
        + r"""

\subsection*{Competing interests}
The author declares that they have no competing interests.

\subsection*{Funding}
This work received no specific external grant funding.

\subsection*{Author's contributions}
GS conceived the mathematical framework, designed and implemented the \texttt{DeepFuzzyTorch} open-source PyTorch software package, executed the multi-cohort neuroimaging, pan-cancer transcriptomic, and pathology foundation-model benchmarks, and wrote and approved the final manuscript.
"""
    )

    gb_full = (
        gb_header
        + "\n"
        + gb_background
        + "\n"
        + gb_implementation
        + "\n"
        + gb_results
        + "\n\\FloatBarrier\n"
        + gb_discussion
        + "\n"
        + gb_conclusions
        + "\n"
        + gb_methods
        + "\n"
        + gb_declarations
        + "\n\\FloatBarrier\n"
        + parts["appendix"]
        + "\n\\end{document}\n"
    )
    gb_full = gb_full.replace(
        "and planned for public release at",
        "and publicly archived under the MIT License at",
    )
    assert_no_empty_same_level_headings(gb_full, "Genome_Biology_Software")
    return gb_full


def build_nature_communications_tex(parts: Dict[str, str]) -> str:
    nc_title_plain = (
        "Reliable Multimodal Biomedical Fusion Under Missing Modalities, Biomarker Discordance, "
        "and High-Dimensional Omics Redundancy via DeepFuzzyTorch"
    )
    nc_preamble = update_hypersetup(
        parts["preamble"].replace(r"\setcounter{dbltopnumber}{2}", r"\setcounter{dbltopnumber}{1}"),
        pdftitle=nc_title_plain,
        pdfauthor="Guanqun Sun",
    )

    nc_abstract_body = (
        r"Multimodal biomedical AI faces missing modalities, biomarker discordance, and opaque "
        r"decisions. We present \texttt{DeepFuzzyTorch}, a curated PyTorch integration of "
        r"fuzzy fusion, rough feature selection, interpretable rules, and test-time adaptation. "
        r"Vacuity mapping guarantees exact $0.00$ fixed-width padding invariance for "
        r"unacquired slots. Across 5,356 participants and 4,849 pathology records, empirical "
        r"benchmarks show ADNI Cohort~1 AUROC of $0.8270$ ($N=1,137$) and $0.8371$ "
        r"(complete $N_{\mathrm{comp}}=854$), missingness retention $\mathrm{MRR}=0.9344$, "
        r"60-month conversion AUROC $0.9419$, and OASIS-3 transfer AUROC $0.8639$. "
        r"Observed conflict detects swaps at $0.7000$ AUROC and is $1.51\times$ higher "
        r"in discordant ADNI participants. A 20-gene TCGA reduct yields $0.7403$ versus "
        r"$0.6925$ AUROC for the matched unreduced model."
    )
    raw_wc, plain_wc = count_abstract_words(nc_abstract_body)
    if raw_wc > 150 or plain_wc > 150 or r"\cite" in nc_abstract_body:
        raise AssertionError(
            f"Nature Communications abstract violates <=150-word unreferenced rule: raw={raw_wc}, plain={plain_wc}"
        )

    nc_header = (
        nc_preamble
        + r"""\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize \textit{Formatted in Nature Portfolio (Results-First) Style}}
\fancyhead[R]{\footnotesize \textit{DeepFuzzyTorch: Reliable Multimodal Biomedical AI Under Missingness \& Conflict}}
\fancyfoot[C]{\footnotesize \thepage}
\renewcommand{\headrulewidth}{0.4pt}

\title{\vspace{-1.0em}\textbf{Reliable Multimodal Biomedical Fusion Under Missing Modalities, Cross-Modal Biomarker Discordance, and High-Dimensional Omics Redundancy via \texttt{DeepFuzzyTorch}}}

\author{
\textbf{Guanqun Sun}\\
\small School of Medical Imaging and Artificial Intelligence, Hangzhou Medical College, Hangzhou 310053, China\\
\small Open-Source PyTorch Operator Library \& Benchmark Suite: \url{https://github.com/SUN-1024/DeepFuzzyTorch}
}

\date{}

\begin{document}

\maketitle

\begin{abstract}
"""
        + nc_abstract_body
        + "\n\\end{abstract}\n"
    )

    # 1. Fold Related Work naturally into Introduction (NO \section{Related Work}, NO \subsection inside Intro)
    related_no_tables, related_tables = extract_tables(parts["related"])
    related_prose = re.sub(
        r"^\\section\{Related Work\}\s*(?:\\label\{[^}]+\}\s*)?",
        "",
        related_no_tables.strip(),
        count=1,
    )
    # Remove standalone Table 1 lead-in sentence if present and replace with smooth Nature transition
    related_prose = re.sub(
        r"Table~\\ref\{tab:paradigm_comparison\}\s+summarizes[^.]*\.\s*",
        "",
        related_prose,
        count=1,
    )
    related_prose = re.sub(r"\\subsection\*?\{[^}]+\}\s*", "\n\n", related_prose).strip()
    related_prose = _replace_ws_insensitive(
        related_prose,
        r"Our Theorem~\ref{thm:degeneration} and Corollary~\ref{cor:e2e_vacuity} extend",
        r"As formalized in Methods (Section~\ref{sec:method}, Theorem~\ref{thm:degeneration} and Corollary~\ref{cor:e2e_vacuity}), our framework extends",
    )
    folded_related = (
        "\n\nPrior multimodal and uncertainty-aware learning frameworks address only isolated facets "
        r"of these clinical failure modes (Supplementary Table~\ref{tab:paradigm_comparison})."
        "\n\\label{sec:related}\n"
        + related_prose
        + "\n\n"
    )

    intro_no_roadmap = strip_cs_roadmap(parts["intro"])
    bridge_anchor = r"To bridge this gap, we present \texttt{DeepFuzzyTorch}"
    if bridge_anchor in intro_no_roadmap:
        idx_b = intro_no_roadmap.index(bridge_anchor)
        nc_intro = intro_no_roadmap[:idx_b].rstrip() + folded_related + intro_no_roadmap[idx_b:].lstrip()
    else:
        nc_intro = intro_no_roadmap.rstrip() + folded_related

    nc_intro = _replace_ws_insensitive(
        nc_intro,
        r"($w_2=0.0000$, Theorem~\ref{thm:degeneration})",
        r"($w_2=0.0000$; Methods, Theorem~\ref{thm:degeneration})",
    )
    nc_intro = _replace_ws_insensitive(
        nc_intro,
        r"We prove and numerically verify five foundational properties and an end-to-end corollary.",
        r"In Methods (Section~\ref{sec:method}) and Supplementary Information (Appendix~\ref{app:proofs}), we prove and numerically verify five foundational properties and an end-to-end corollary.",
    )

    nc_intro = (
        nc_intro.rstrip()
        + "\n\nFormal operator definitions and full statements of our five mathematical guarantees "
        r"(Theorems~\ref{thm:closure}--\ref{thm:rough} and Corollary~\ref{cor:e2e_vacuity}) are presented in Methods "
        r"(Section~\ref{sec:method}), with end-to-end forward inference pseudocode "
        r"(Supplementary Algorithm~\ref{alg:deepfuzzy}), supplementary tables, complete proofs, and cohort provenance "
        r"provided in Supplementary Information (Appendices~\ref{app:supp_tables}--\ref{app:provenance})." + "\n"
    )

    # 2. Results (qualify forward Theorem reference on first Results appearance & enforce <= 10 display items)
    nc_results_raw = re.sub(
        r"^\\section\{(?:Experimental )?Results\}",
        r"\\section{Results}",
        parts["results"],
        count=1,
    )
    nc_results_raw = _replace_ws_insensitive(
        nc_results_raw,
        r"Figure~\ref{fig:theorems} summarizes the empirical verification of Theorems~\ref{thm:closure}--\ref{thm:rough}",
        r"Figure~\ref{fig:theorems} summarizes the empirical verification of the five mathematical properties formalized in Methods (Section~\ref{sec:method}, Theorems~\ref{thm:closure}--\ref{thm:rough})",
    )

    # 3. Discussion (fold concluding synthesis cleanly into Discussion — NEVER into Data/Code Availability)
    nc_discussion = (
        parts["discussion"].rstrip()
        + "\n\n\\noindent\\textbf{Concluding synthesis.} "
        + parts["conc_synthesis"]
        + "\n"
    )

    # 4. Methods (merge Methodology + Experimental Setup WITHOUT empty orphan subsection 5.7!)
    exp_without_sec = re.sub(
        r"^\\section\{Experimental Setup\}\s*",
        "",
        parts["experiments"].strip(),
        count=1,
    )
    # If \label{sec:experiments} was right under \section{Experimental Setup}, tuck it inside the first \subsection
    exp_label_m = re.match(r"^(\\label\{[^}]+\})\s*(\\subsection\*?\{[^}]+\})", exp_without_sec)
    if exp_label_m:
        exp_without_sec = (
            f"{exp_label_m.group(2)}\n{exp_label_m.group(1)}"
            + exp_label_m.end() * ""
            + exp_without_sec[exp_label_m.end() :]
        )

    nc_methods_raw = (
        parts["method"].replace(r"\section{Methodology}", r"\section{Methods}", 1).rstrip()
        + "\n\n"
        + exp_without_sec
        + "\n"
    )

    # 5. Enforce Nature Communications main-text float cap (<= 10 total floats: figures + tables + algorithms)
    nc_methods_no_tab, method_tables = extract_tables(nc_methods_raw)
    nc_methods, method_algs = extract_algorithms(nc_methods_no_tab)
    n_main_figs = len(
        re.findall(r"\\begin\{figure\*?\}", nc_intro + "\n" + nc_results_raw + "\n" + nc_discussion + "\n" + nc_methods)
    )
    max_main_tables = max(0, 10 - n_main_figs)
    preferred_main_tables = {"tab:adni_main", "tab:tcga_fm"}
    nc_results, results_moved_tables, kept_main_table_labels = partition_tables_for_nc(
        nc_results_raw, keep_labels=preferred_main_tables, max_keep=max_main_tables
    )

    supp_tables: List[Tuple[str, str]] = related_tables + results_moved_tables + method_tables
    moved_labels = {lbl for lbl, _ in supp_tables if lbl}
    moved_alg_labels = {lbl for lbl, _ in method_algs if lbl}

    # Prefix standalone Table~\ref{<moved_lbl>} and Algorithm~\ref{<moved_alg_lbl>} in main text with "Supplementary "
    def _prefix_supp_refs(body: str) -> str:
        out = body
        for lbl in sorted(moved_labels):
            out = re.sub(
                rf"(?<!Supplementary )(?<!--)Table~\\ref\{{{re.escape(lbl)}\}}",
                rf"Supplementary Table~\\ref{{{lbl}}}",
                out,
            )
        for lbl in sorted(moved_alg_labels):
            out = re.sub(
                rf"(?<!Supplementary )(?<!--)Algorithm~\\ref\{{{re.escape(lbl)}\}}",
                rf"Supplementary Algorithm~\\ref{{{lbl}}}",
                out,
            )
        return out

    nc_intro = _prefix_supp_refs(nc_intro)
    nc_results = _prefix_supp_refs(nc_results)
    nc_discussion = _prefix_supp_refs(nc_discussion)
    nc_methods = _prefix_supp_refs(nc_methods)

    main_body_pre_appendix = nc_intro + "\n" + nc_results + "\n\\FloatBarrier\n" + nc_discussion + "\n" + nc_methods
    total_main_figs = len(re.findall(r"\\begin\{figure\*?\}", main_body_pre_appendix))
    total_main_tabs = len(re.findall(r"\\begin\{table\*?\}", main_body_pre_appendix))
    total_main_algs = len(re.findall(r"\\begin\{algorithm\*?\}", main_body_pre_appendix))
    total_main_display = total_main_figs + total_main_tabs + total_main_algs
    if total_main_display > 10:
        raise AssertionError(
            f"Nature Communications main-text floats ({total_main_display} = {total_main_figs} figs + "
            f"{total_main_tabs} tables + {total_main_algs} algorithms) exceed strict <= 10 limit!"
        )

    # 6. Dedicated Nature Portfolio Backmatter Sections (Data Availability, Code Availability, Author Contributions, Competing Interests)
    nc_backmatter = r"""\section*{Data Availability}
All multi-center clinical, neuroimaging, multi-omics, and histopathology benchmarks analyzed in this study derive from publicly accessible cohorts: the Alzheimer's Disease Neuroimaging Initiative (ADNI Cohorts~1, 2, and Longitudinal; \url{https://adni.loni.usc.edu/}), OASIS-3 (\url{https://www.oasis-brains.org/}), the NCI Genomic Data Commons and cBioPortal TCGA PanCancer Atlas across seven cancer cohorts (\url{https://www.cbioportal.org/}), the Reactome Pathway Knowledgebase (\url{https://reactome.org/}), Camelyon17, and Zenodo NCT-CRC-HE-100K / CRC-VAL-HE-7K. Because participant-level ADNI and OASIS-3 tables cannot be redistributed directly under third-party Data Use Agreements, local preprocessing scripts consume harmonized credentialed CSV exports and pre-extracted feature caches rather than bundling raw LONI neuroimaging or raw WSI pipelines, while aggregated quantitative summary metrics (\texttt{results/\allowbreak benchmark\_\allowbreak results.json}), custom-background Reactome enrichment tables (\texttt{results/\allowbreak tcga\_\allowbreak top20\_\allowbreak reactome\_\allowbreak enrichment.json}), Markdown summary tables, and cryptographic SHA-256 digests (\texttt{results/\allowbreak reproducibility\_\allowbreak manifest.json}) are bundled in the public reproducibility repository.

\section*{Code Availability}
The complete \texttt{DeepFuzzyTorch} (\texttt{v0.2.1}) PyTorch operator library, all baseline implementations (including \texttt{QMF (adapted)}, \texttt{DDEF (adapted)}, \texttt{FUML (adapted)}, \texttt{RCML (adapted)}/\texttt{DrFuse (adapted)}, \texttt{XGBoost}, \texttt{CatBoost}, and \texttt{TabPFN~v2}), automated multi-track benchmark and runtime profiling scripts (\texttt{benchmarks/*.py}), mathematical verification test suite (\texttt{tests/\allowbreak test\_\allowbreak operators\_\allowbreak and\_\allowbreak theorems.py}), synthetic quickstart demo (\texttt{examples/\allowbreak quickstart\_\allowbreak synthetic\_\allowbreak demo.py}), and one-click figure/table generators are released under the MIT License at \url{https://github.com/SUN-1024/DeepFuzzyTorch}.

\section*{Author Contributions}
G.S. conceived the mathematical framework, designed and implemented the \texttt{DeepFuzzyTorch} PyTorch operator library, curated and audited the multi-center neuroimaging, pan-cancer transcriptomic, and computational pathology benchmarks, performed the statistical analyses, and wrote the manuscript.

\section*{Competing Interests}
The author declares no competing interests.
"""

    # 7. Supplementary Information (\appendix): Supplementary Algorithm & Tables + Proofs & Cohort Provenance
    supp_tables_sec = ""
    if method_algs or supp_tables:
        rendered_algs = "\n\n".join(alg for _, alg in method_algs)
        rendered_supp_tables = "\n\n".join(tbl for _, tbl in supp_tables)
        supp_blocks = [b for b in (rendered_algs, rendered_supp_tables) if b]
        supp_tables_sec = (
            "\\section{Supplementary Algorithm and Tables}\n"
            "\\label{app:supp_tables}\n"
            "\\renewcommand{\\thealgorithm}{S\\arabic{algorithm}}\n"
            "\\setcounter{algorithm}{0}\n"
            "\\renewcommand{\\thetable}{S\\arabic{table}}\n"
            "\\setcounter{table}{0}\n"
            "Supplementary Algorithm~S1 and Supplementary Tables~S1--S"
            + str(len(supp_tables))
            + " provide end-to-end forward inference pseudocode, conceptual paradigm comparisons, "
            "component and supervision-fairness ablations, combinatorial missing-modality breakdowns, "
            "longitudinal/external/discordance evaluations, and computational scaling profiles referenced in the main text.\n\n"
            + "\n\n".join(supp_blocks)
            + "\n\\FloatBarrier\n\n"
        )

    app_rest = re.sub(r"^\\appendix\s*", "", parts["appendix"], count=1)
    nc_appendix = "\\FloatBarrier\n\\appendix\n" + supp_tables_sec + app_rest

    nc_full = (
        nc_header
        + "\n"
        + main_body_pre_appendix
        + "\n\\FloatBarrier\n"
        + nc_backmatter
        + "\n"
        + nc_appendix
        + "\n\\end{document}\n"
    )
    nc_full = nc_full.replace(
        "and planned for public release at",
        "and publicly archived under the MIT License at",
    )

    # Programmatic structural compliance assertions for Nature Communications
    for forbidden_sec in ("Related Work", "Experimental Results", "Experimental Setup", "Conclusion", "Conclusions"):
        if re.search(rf"\\section\*?\{{{re.escape(forbidden_sec)}\}}", nc_full):
            raise AssertionError(f"[Nature_Communications] Forbidden top-level section found: {forbidden_sec}")
    assert_no_empty_same_level_headings(nc_full, "Nature_Communications")
    print(
        f"[NC Verification] Abstract words: raw={raw_wc}, plain={plain_wc} (<=150) | "
        f"Main floats: {total_main_display}/10 ({total_main_figs} figs + {total_main_tabs} tables {kept_main_table_labels} + {total_main_algs} algs) | "
        f"Supplementary algs: {len(method_algs)}, tables: {len(supp_tables)}"
    )
    return nc_full


_PROTECTED_LATEX_PHRASES: List[str] = sorted(
    {
        r"HesitantFuzzyAggregator} parameters are trained via Adam ($\text{lr}=0.025$",
        r"Our Theorem~\ref{thm:degeneration} and Corollary~\ref{cor:e2e_vacuity} extend",
        r"DeepFuzzyTorch: An Open-Source Differentiable Fuzzy-Rough PyTorch Library",
        r"without claiming statistically significant triage separation",
        r"The remainder of this article is organized as follows.",
        r"The remainder of this paper is organized as follows.",
        r"transductive unsupervised $5,000$-gene MAD cache",
        r"EarlyConcat-MLP (4-Mod, \texttt{All-1000}",
        r"DrFuse (adapted, \textit{AAAI'24}, 4-Mod",
        r"RCML (adapted, \textit{AAAI'24}, 4-Mod",
        r"DempsterShafer (\textit{IJAR'19}, 4-Mod",
        r"EvidentialDL (\textit{TPAMI'23}, 4-Mod",
        r"and publicly archived under the MIT License at",
        r"inner-split composite objective peak",
        r"c_0 := \frac{1}{2}\bar{g}_{m^*} > 0",
        r"150 Adam steps ($\text{lr}=0.025$)",
        r"\textbf{\texttt{DeepFuzzyTorch} (Ours)}",
        r"EarlyConcat-MLP (4-Mod, $K^*=20$)",
        r"GatedAttention (4-Mod, $K^*=20$)",
        r"w/o Conflict Hesitancy Discount",
        r"DrFuse (\textit{AAAI'24}, adapt.)",
        r"RCML (\textit{AAAI'24}, adapt.)",
        r"and planned for public release at",
        r"w/o White-Box Concept TSK Rule Head",
        r"Macro-Average (15 Subsets)",
        r"Incomplete MRR (14 Subsets)",
        r"w/o Non-Additive Choquet",
        r"GatedAttention-Transformer",
        r"QMF (\textit{ICML'23}, adapt., 4-Mod",
        r"Matched \texttt{All-1000}",
        r"Apple Metal MPS supported",
        r"ECE $0.0461$ vs.\ $0.0651$",
        r"Joint Cold, No Brier/Aux",
        r"320 ROIs; No Reducer",
        r"Shared Fused AuxWeight",
        r"NCT-CRC Binary (\texttt{Discovery}",
        r"Dempster--Shafer Fusion",
        r"w/o Fuzzy-Rough Gate",
        r"w/o Vacuity Masking",
        r"Complete source code",
        r"Camelyon17 ($N=2,500$",
        r"Camelyon17 ($n=1,000$",
        r"EarlyConcat-MLP (",
        r"LateAvg-Ensemble (",
        r"NCT-CRC Binary ($n=1,350$",
        r"EvidentialDL (TMC",
        r"peaking at $k=12$",
        r"peaking at $k=28$",
        r"DrFuse (adapted",
        r"RCML (adapted)",
        r"RCML (adapted",
        r"All source code",
        r"(a+c_0)^p - a^p",
        r"v_i / \bar{v}",
        r"1 - 2^{-24}",
        r"QMF (Zhang et al., \textit{ICML'23}, adapt.)",
        r"QMF (\textit{ICML'23}, adapt.",
        r"QMF (adapted",
        r"NCT-CRC Binary (",
        r"QMF (Zhang",
        r"Full Model",
    },
    key=len,
    reverse=True,
)


def _split_long_cite_token(tok: str, max_len: int = 70) -> List[str]:
    m = re.match(r"^(.*?)(\\cite\{[^}]+\})(.*)$", tok)
    if not m:
        return [tok]
    prefix, cite_cmd, suffix = m.group(1), m.group(2), m.group(3)
    keys = cite_cmd[6:-1].split(",")
    chunks: List[str] = []
    cur = f"{prefix}\\cite{{"
    for idx, key in enumerate(keys):
        sep = "," if idx < len(keys) - 1 else "}" + suffix
        piece = key + sep
        if idx > 0 and len(cur) + len(piece) > max_len:
            chunks.append(cur + "%")
            cur = piece
        else:
            cur += piece
    chunks.append(cur)
    return chunks


def _break_long_token(tok: str, limit: int) -> List[str]:
    if "\\cite{" in tok and len(tok) > 65:
        return _split_long_cite_token(tok, max_len=min(70, limit - 10))
    m_img = re.match(r"^(\\includegraphics\[[^\]]*\]\{\.\./figures/Fig[1-9]_)(.+)$", tok)
    if m_img and len(tok) > 80:
        return [m_img.group(1) + "%", m_img.group(2)]
    return [tok]


def wrap_latex_lines_95(tex: str, limit: int = 95) -> str:
    """Wrap LaTeX source lines exceeding ``limit`` chars cleanly without breaking TeX semantics or audit keys."""
    out_lines: List[str] = []
    for raw_line in tex.splitlines():
        if len(raw_line) <= limit:
            out_lines.append(raw_line)
            continue

        indent_len = len(raw_line) - len(raw_line.lstrip(" "))
        indent = raw_line[:indent_len]
        body = raw_line[indent_len:]

        # Insert harmless breakpoints before long protected headers/formulas
        if body.startswith(r"\title{\vspace{-1.0em}\textbf{"):
            out_lines.append(indent + r"\title{\vspace{-1.0em}%")
            body = body[len(r"\title{\vspace{-1.0em}") :]
        elif body.startswith(r"pdftitle={"):
            out_lines.append(indent + r"pdftitle={%")
            body = body[len(r"pdftitle={") :]
        body = re.sub(r"\)\+([0-9])", r") + \1", body)
        body = body.replace(r"\big(\mathcal{A}", r"\big( \mathcal{A}")

        masked = body
        for phrase in _PROTECTED_LATEX_PHRASES:
            if phrase in masked:
                masked = masked.replace(phrase, phrase.replace(" ", "\x01"))

        raw_tokens = [t for t in masked.split(" ") if t != ""]
        tokens: List[str] = []
        for t in raw_tokens:
            if "\x01" not in t:
                tokens.extend(_break_long_token(t, limit))
            else:
                tokens.append(t)

        cont_indent = indent if indent_len <= 4 else ""
        wrapped_here: List[str] = []
        cur_line = indent
        at_line_start = True

        for tok in tokens:
            if cur_line.endswith("%"):
                wrapped_here.append(cur_line)
                cur_line = ""
                at_line_start = True

            if at_line_start:
                prefix = indent if not wrapped_here else (cont_indent if len(cont_indent) + len(tok) <= limit else "")
                cur_line = prefix + tok
                at_line_start = False
            elif len(cur_line) + 1 + len(tok) <= limit:
                cur_line = cur_line + " " + tok
            else:
                wrapped_here.append(cur_line)
                prefix = cont_indent if len(cont_indent) + len(tok) <= limit else ""
                cur_line = prefix + tok
            if cur_line.endswith("%"):
                wrapped_here.append(cur_line)
                cur_line = ""
                at_line_start = True

        if cur_line:
            wrapped_here.append(cur_line)

        for wl in wrapped_here:
            restored = wl.replace("\x01", " ")
            if len(restored) > limit:
                raise AssertionError(f"Line exceeded {limit} chars ({len(restored)}): {restored}")
            out_lines.append(restored)

    return "\n".join(out_lines) + ("\n" if tex.endswith("\n") else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Genome Biology & Nature Communications venue templates.")
    parser.add_argument("--source", default=DEFAULT_INF_FUS_TEX, help="Source Information Fusion .tex path")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for generated .tex files")
    parser.add_argument("--wrap-source", action="store_true", help="Also wrap the source IF .tex file <= 95 chars")
    args = parser.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        src = f.read()

    parts = parse_if_sections(src)
    os.makedirs(args.out_dir, exist_ok=True)
    gb_path = os.path.join(args.out_dir, "manuscript_Genome_Biology_Software.tex")
    nc_path = os.path.join(args.out_dir, "manuscript_Nature_Communications.tex")

    gb_tex = wrap_latex_lines_95(build_genome_biology_tex(parts), limit=95)
    nc_tex = wrap_latex_lines_95(build_nature_communications_tex(parts), limit=95)

    if args.wrap_source:
        wrapped_src = wrap_latex_lines_95(src, limit=95)
        with open(args.source, "w", encoding="utf-8") as f:
            f.write(wrapped_src)

    with open(gb_path, "w", encoding="utf-8") as f:
        f.write(gb_tex)
    with open(nc_path, "w", encoding="utf-8") as f:
        f.write(nc_tex)

    print(f"Wrote {gb_path} and {nc_path}")


if __name__ == "__main__":
    main()
