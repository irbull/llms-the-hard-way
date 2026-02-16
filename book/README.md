# Book

This directory contains the [mdBook](https://rust-lang.github.io/mdBook/) source for the tutorial.

## Prerequisites

Install mdBook:

```bash
cargo install mdbook
```

## Development

Serve the book locally with live reload:

```bash
cd book
mdbook serve --open
```

This starts a local server at `http://localhost:3000` and opens it in your browser. Pages reload automatically as you edit files in `src/`.

## Build

To build static HTML:

```bash
cd book
mdbook build
```

Output goes to `book/build/`.
