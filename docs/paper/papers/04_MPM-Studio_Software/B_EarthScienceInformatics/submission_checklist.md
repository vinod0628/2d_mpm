# Submission checklist — Earth Science Informatics (Springer Nature), Software Article

Manuscript: `manuscript.docx` (built from `papers/_build/content/studio_esi.js` and `_studio.js`; rebuild instead of editing numbers).

Journal: https://link.springer.com/journal/12145 — submission guidelines: https://link.springer.com/journal/12145/submission-guidelines
(the page redirects automated clients to a sign-in, so confirm the items below in a browser).

Applied in this version:
- Article type: Software Article — "Design and implementation" between Introduction and Results, "Software files" subsection,
  "Availability and requirements" after the Conclusions (project name, home page, operating system, language, other requirements,
  licence, restrictions for non-academics).
- Abstract 150–250 words without references; 4–6 keywords.
- Author–year citations; alphabetical reference list with DOIs as links.
- Figures "Fig. n" with captions; figure files in `figures/` (Fig_1 …); tables with captions above; continuous line numbering.
- "Declarations" before the references (Funding, Competing interests, Data availability, Code availability, Author contributions).
- Use of a large language model documented in the Design and implementation section (Springer Nature policy), not as an author.
- Publishing model: hybrid — no charge on the standard (subscription) route; open access (Open Choice) optional.

To complete:
- [ ] Software must be freely available for non-commercial use: make the repository public before submission.
- [ ] Replace the placeholder https://github.com/[account]/mpm-studio and [Zenodo DOI of the release] (Zenodo release DOI).
- [ ] Choose the licence (PyQt5 is GPL v3 or commercial; a GPL v3 release is the usual choice).
- [ ] Authors, affiliations, e-mail, ORCID; acknowledgements; funding; author contributions.
- [ ] Suggested reviewers in the cover letter.
- [ ] Run the full test suite (`python -m pytest python/tests`) on the release commit and report the result in the README.
