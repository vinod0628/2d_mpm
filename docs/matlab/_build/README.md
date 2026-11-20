# Development-record documents: source

The Word files in `docs/matlab/` (`M0_Common_Framework.docx`, `M1_…` to `M7_…`) are generated from
the content files in `content/`, one per document. Edit the content file, not the .docx, and rebuild.

```
npm install docx@9          # once, in any folder; point NODE_PATH at its node_modules
node build.js               # all documents
node build.js m7_anomaly    # one document
```

- `lib.js`: page layout, tables, lists and a small math markup that becomes native Word equations
  (see the header of `lib.js`); equations are numbered automatically and cited as `[[id]]`.
- `refs.js`: the shared reference list (Elsevier / Harvard style).
- New method M<n>: copy a content file to `content/m<n>_<name>.js`, keep the section order
  (Overview, Theory, Mathematical formulation, Implementation, Verification and demo results, References).

This folder is documentation tooling only; the MATLAB tool does not use it.
