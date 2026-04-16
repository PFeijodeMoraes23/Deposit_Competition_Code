BASELINE_PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage[letterpaper, margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[english]{babel}
\usepackage[dvipsnames]{xcolor}

% Math and fonts
\usepackage{amssymb, mathrsfs}
\usepackage{amsthm}
\usepackage{mathtools}
\usepackage{dsfont}
\usepackage[normalem]{ulem}
\usepackage{relsize}
\usepackage{array}

% Graphics and tables
\usepackage{graphicx}  
\usepackage{float}
\usepackage{xltabular}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\usepackage{xurl}
\usepackage{subcaption}

% Spacing and formatting
\setlength{\parindent}{1cm}
\usepackage{setspace}
\usepackage{indentfirst}
\usepackage{titlesec}
\usepackage{multirow}

% Lists
\usepackage{enumitem}

% Bibliography
\usepackage[backend=biber, style=authoryear, maxcitenames=2, sortcites=true, sorting=ynt]{biblatex}
\DeclareDelimFormat[bib,textcite,parencite]{finalnamedelim}{\addspace\&\addspace}
\DeclareDelimFormat[bib,parencite]{nameyeardelim}{\addcomma\space}
\DeclareDelimFormat[textcite]{nameyeardelim}{\addspace}
\addbibresource{../References.bib}

\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\cmarkbold}{\ding{52}}

% TikZ and PGFPlots
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepgfplotslibrary{fillbetween}
\usepackage{footmisc}
\usepackage[font=small,labelfont=bf]{caption}

% Custom commands
\newcommand{\linearmap}[3]{#1:#2\to#3}
\newcommand{\myarrow}[0]{\,\to\,}
\newcommand{\mycomma}[0]{\,,\,}
\newcommand{\myspace}[0]{\,\quad\,}
\newcommand{\ida}[0]{\left(\Rightarrow\right)}
\newcommand{\volta}[0]{\left(\Leftarrow\right)}
\newcommand{\myiff}[0]{\,\longleftrightarrow\,}
\newcommand{\tabspace}[0]{\myspace\myspace\myspace}
\newcommand{\textbfit}[1]{\text{\textbf{\textit{#1}}}}
\newcommand{\openball}[3][]{\mathrm{B}_{#1}\left(#2;#3\right)}

% Calculus
\newcommand{\derivative}[2]{\frac{\mathrm{d} #1}{\mathrm{d} #2}}
\newcommand{\partialderivative}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\mylog}[1]{\ln{\left( #1 \right)}}
\newcommand{\mydet}[1]{\mathrm{det}{\left(#1\right)}}
\newcommand{\dirderi}[2]{\mathrm{D}_{\bold{#1}}{#2}}
\newcommand{\minimize}[2][ ]{\underset{#1}{\mathrm{min}} \myspace{#2}}
\newcommand{\maximize}[2][ ]{\underset{#1}{\mathrm{max}} \myspace{#2}}
\newcommand{\subjectto}[1]{\mathrm{s.t.}\myspace{#1}}

% Proper argmax/argmin definitions
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}

% Sets
\newcommand{\nnatural}[0]{\mathbb{N}}
\newcommand{\real}[0]{\mathbb{R}}
\newcommand{\zhail}[0]{\mathbb{Z}}
\newcommand{\rational}[0]{\mathbb{Q}}
\newcommand{\powerset}[1]{\mathbb{P}\left(#1\right)}

% Probability and statistics
\newcommand{\expectedvalue}[2][]{\mathbb{E}_{#1}\left[#2\right]}
\newcommand{\condexp}[3][]{\mathbb{E}_{#1}\left[#2\middle|#3\right]}
\newcommand{\indicator}[1]{\boldsymbol{1}\left\{#1\right\}}
\newcommand{\condprob}[2]{P\left(#1 \mid #2 \right)}
\newcommand{\plimarrow}[0]{\overset{p}{\longrightarrow}}
\newcommand{\convdistr}[0]{\overset{d}{\longrightarrow}}
\newcommand{\indep}[0]{\mathrel{\perp\!\!\!\perp}}

% Theorems and definitions
\theoremstyle{plain}
\newtheorem{theo}{Theorem}[subsection]
\newtheorem{prop}{Proposition}[subsection]
\newtheorem{deff}{Definition}[subsection]
\newtheorem{lemma}{Lemma}[subsection]

% Custom environments (with and without *)
\newtheorem*{theo*}{Theorem}
\newtheorem*{prop*}{Proposition}
\newtheorem*{deff*}{Definition}
\newtheorem*{lemma*}{Lemma}
\newcommand{\PFMcomment}[1]{\textcolor{MidnightBlue}{#1}}

% Allow display breaks
\allowdisplaybreaks

% Hyperlinks (loaded last)
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=RoyalBlue, urlcolor=RoyalBlue, citecolor=RoyalBlue}
\urlstyle{tt}

\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\setlength{\emergencystretch}{2em}
\AtBeginEnvironment{xltabular}{\footnotesize}
\AtBeginEnvironment{tabularx}{\footnotesize}
\AtBeginEnvironment{longtable}{\footnotesize}

% Table packages:
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{pdflscape}
\usepackage{microtype}
\usepackage{csquotes}
\usepackage{etoolbox}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{listings}
\begin{document}
"""

BASELINE_POSTAMBLE = r"""\end{document}
"""

DRAFTS_DIR = r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition"

def wrap_table(latex_table_str: str) -> str:
    """Wraps a raw latex table string inside the baseline preamble document wrapper."""
    return BASELINE_PREAMBLE + "\n" + latex_table_str + "\n" + BASELINE_POSTAMBLE
