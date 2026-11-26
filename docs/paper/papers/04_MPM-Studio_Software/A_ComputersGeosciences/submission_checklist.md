# Submission checklist — Computers & Geosciences (Elsevier)

Manuscript: `manuscript.docx` (built from `papers/_build/content/studio_cg.js` and `_studio.js`; rebuild instead of editing numbers).

Journal: https://www.sciencedirect.com/journal/computers-and-geosciences — guide for authors linked from that page (Elsevier pages
refuse automated clients, so confirm the items below in a browser).

Applied in this version:
- Article type: Research paper. Highlights: 5 bullets, each at most 85 characters (also to be uploaded as a separate file).
- Abstract at most 250 words; 1–7 keywords.
- Author–year (Harvard) citations; alphabetical reference list with DOIs; software cited with [software] tag.
- Numbered sections; figures "Fig. n" with captions; tables with captions above; continuous line numbers; double spacing.
- "Computer code availability" section with name, developers/contact, year, hardware, software, language, size, licence and link.
- CRediT authorship statement; Declaration of competing interest; Declaration of generative AI; Data availability.

Mandatory for C&G (scope and code policy, checked 24 November 2026):
- [ ] **Public repository at submission** (GitHub or similar): licence file, English README with installation steps, dependency list,
      the data needed to reproduce the main results (demo projects + Toodoggone data), user guide; no single zip file.
- [ ] Replace the placeholder https://github.com/[account]/mpm-studio and [Zenodo DOI of the release] (Zenodo release DOI) in the manuscript and the cover letter.
- [ ] Choose the licence. PyQt5 is GPL v3 (or commercial), so a public release normally has to be GPL v3 compatible.
- [ ] Word count: C&G research papers are limited to about 5,500 words (6,500 at revision, ±10 %) excluding abstract, references,
      captions and tables — shorten Section 3 if needed (see the word count printed by the build script).
- [ ] The scope page warns against papers that are only an interface or packaging of known methods. Keep the emphasis on the
      plug-in architecture, the common evaluation layer, verification and the benchmark (cover letter paragraph 3).

To complete:
- [ ] Authors, affiliations, e-mail, ORCID; acknowledgements and funding; CRediT roles.
- [ ] Suggested reviewers in the cover letter.
- [ ] Highlights as a separate file (copy from the manuscript).
- [ ] Run the full test suite (`python -m pytest python/tests`) on the release commit and report the result in the README.
