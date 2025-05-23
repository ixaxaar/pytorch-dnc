<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Change Log](#change-log)
  - [Unreleased](#unreleased)
  - [1.0.1 (2019-04-05)](#101-2019-04-05)
  - [1.0.0 (2019-04-05)](#100-2019-04-05)
  - [0.0.9 (2018-04-23)](#009-2018-04-23)
  - [0.0.7 (2017-12-20)](#007-2017-12-20)
  - [0.0.6 (2017-11-12)](#006-2017-11-12)
  - [0.5.0 (2017-11-01)](#050-2017-11-01)
  - [0.0.3 (2017-10-27)](#003-2017-10-27)
  - [0.0.2 (2017-10-26)](#002-2017-10-26)
  - [v0.0.1 (2017-10-26)](#v001-2017-10-26)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Change Log

## [Unreleased](https://github.com/ixaxaar/pytorch-dnc/tree/HEAD)

[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/1.0.1...HEAD)

**Merged pull requests:**

- Fixes for \#43 [\#44](https://github.com/ixaxaar/pytorch-dnc/pull/44) ([ixaxaar](https://github.com/ixaxaar))

## [1.0.1](https://github.com/ixaxaar/pytorch-dnc/tree/1.0.1) (2019-04-05)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/1.0.0...1.0.1)

**Closed issues:**

- When running adding task -- ModuleNotFoundError: No module named 'index'  [\#39](https://github.com/ixaxaar/pytorch-dnc/issues/39)
- SyntaxError [\#36](https://github.com/ixaxaar/pytorch-dnc/issues/36)
- PySide dependency error [\#33](https://github.com/ixaxaar/pytorch-dnc/issues/33)
- Issues when using pytorch 0.4 [\#31](https://github.com/ixaxaar/pytorch-dnc/issues/31)
- TypeError: cat received an invalid combination of arguments - got \(list, int\), but expected one of: [\#29](https://github.com/ixaxaar/pytorch-dnc/issues/29)

**Merged pull requests:**

- Fixes \#36 and \#39 [\#42](https://github.com/ixaxaar/pytorch-dnc/pull/42) ([ixaxaar](https://github.com/ixaxaar))

## [1.0.0](https://github.com/ixaxaar/pytorch-dnc/tree/1.0.0) (2019-04-05)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/0.0.9...1.0.0)

**Closed issues:**

- Question about the running speed of Pyflann and Faiss  for the SAM model [\#40](https://github.com/ixaxaar/pytorch-dnc/issues/40)
- SyntaxError [\#37](https://github.com/ixaxaar/pytorch-dnc/issues/37)
- Values in hidden become nan [\#35](https://github.com/ixaxaar/pytorch-dnc/issues/35)
- faiss error [\#32](https://github.com/ixaxaar/pytorch-dnc/issues/32)

**Merged pull requests:**

- Port to pytorch 1.x [\#41](https://github.com/ixaxaar/pytorch-dnc/pull/41) ([ixaxaar](https://github.com/ixaxaar))
- fix parens in example usage and gpu usage for SAM [\#30](https://github.com/ixaxaar/pytorch-dnc/pull/30) ([kierkegaard13](https://github.com/kierkegaard13))

## [0.0.9](https://github.com/ixaxaar/pytorch-dnc/tree/0.0.9) (2018-04-23)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/0.0.7...0.0.9)

**Fixed bugs:**

- Use usage vector to determine least recently used memory [\#26](https://github.com/ixaxaar/pytorch-dnc/issues/26)
- Store entire memory after memory limit is reached [\#24](https://github.com/ixaxaar/pytorch-dnc/issues/24)

**Merged pull requests:**

- memory.py: fix indexing for read\_modes transform [\#28](https://github.com/ixaxaar/pytorch-dnc/pull/28) ([jbinas](https://github.com/jbinas))
- Bugfixes [\#27](https://github.com/ixaxaar/pytorch-dnc/pull/27) ([ixaxaar](https://github.com/ixaxaar))

## [0.0.7](https://github.com/ixaxaar/pytorch-dnc/tree/0.0.7) (2017-12-20)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/0.0.6...0.0.7)

**Implemented enhancements:**

- GPU kNNs [\#21](https://github.com/ixaxaar/pytorch-dnc/issues/21)
- Implement temporal addressing for SDNCs [\#18](https://github.com/ixaxaar/pytorch-dnc/issues/18)
- Feature: Sparse Access Memory [\#4](https://github.com/ixaxaar/pytorch-dnc/issues/4)
- SAMs [\#22](https://github.com/ixaxaar/pytorch-dnc/pull/22) ([ixaxaar](https://github.com/ixaxaar))
- Temporal links for SDNC [\#19](https://github.com/ixaxaar/pytorch-dnc/pull/19) ([ixaxaar](https://github.com/ixaxaar))
- SDNC [\#16](https://github.com/ixaxaar/pytorch-dnc/pull/16) ([ixaxaar](https://github.com/ixaxaar))

**Merged pull requests:**

- Add more tasks [\#23](https://github.com/ixaxaar/pytorch-dnc/pull/23) ([ixaxaar](https://github.com/ixaxaar))
- Scale interface vectors, dynamic memory pass [\#17](https://github.com/ixaxaar/pytorch-dnc/pull/17) ([ixaxaar](https://github.com/ixaxaar))
- Update README.md [\#14](https://github.com/ixaxaar/pytorch-dnc/pull/14) ([MaxwellRebo](https://github.com/MaxwellRebo))

## [0.0.6](https://github.com/ixaxaar/pytorch-dnc/tree/0.0.6) (2017-11-12)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/0.5.0...0.0.6)

**Implemented enhancements:**

- Re-write allocation vector code, use pytorch's cumprod [\#13](https://github.com/ixaxaar/pytorch-dnc/issues/13)

**Fixed bugs:**

- Stacked DNCs forward pass wrong [\#12](https://github.com/ixaxaar/pytorch-dnc/issues/12)
- Temporal debugging of memory [\#11](https://github.com/ixaxaar/pytorch-dnc/pull/11) ([ixaxaar](https://github.com/ixaxaar))

## [0.5.0](https://github.com/ixaxaar/pytorch-dnc/tree/0.5.0) (2017-11-01)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/0.0.3...0.5.0)

**Implemented enhancements:**

- Multiple hidden layers per controller layer [\#7](https://github.com/ixaxaar/pytorch-dnc/issues/7)
- Vizdom integration and fix cumprod bug \#5 [\#6](https://github.com/ixaxaar/pytorch-dnc/pull/6) ([ixaxaar](https://github.com/ixaxaar))

**Fixed bugs:**

- Use shifted cumprods, emulate tensorflow's cumprod with exclusive=True [\#5](https://github.com/ixaxaar/pytorch-dnc/issues/5)
- Vizdom integration and fix cumprod bug \\#5 [\#6](https://github.com/ixaxaar/pytorch-dnc/pull/6) ([ixaxaar](https://github.com/ixaxaar))

**Closed issues:**

- Write unit tests [\#8](https://github.com/ixaxaar/pytorch-dnc/issues/8)
- broken links [\#3](https://github.com/ixaxaar/pytorch-dnc/issues/3)

**Merged pull requests:**

- Test travis build [\#10](https://github.com/ixaxaar/pytorch-dnc/pull/10) ([ixaxaar](https://github.com/ixaxaar))
- Implement Hidden layers, small enhancements, cleanups [\#9](https://github.com/ixaxaar/pytorch-dnc/pull/9) ([ixaxaar](https://github.com/ixaxaar))

## [0.0.3](https://github.com/ixaxaar/pytorch-dnc/tree/0.0.3) (2017-10-27)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/0.0.2...0.0.3)

**Implemented enhancements:**

- Implementation of Dropout for controller [\#2](https://github.com/ixaxaar/pytorch-dnc/pull/2) ([ixaxaar](https://github.com/ixaxaar))
- Fix size issue for GRU and vanilla RNN [\#1](https://github.com/ixaxaar/pytorch-dnc/pull/1) ([ixaxaar](https://github.com/ixaxaar))

## [0.0.2](https://github.com/ixaxaar/pytorch-dnc/tree/0.0.2) (2017-10-26)
[Full Changelog](https://github.com/ixaxaar/pytorch-dnc/compare/v0.0.1...0.0.2)

## [v0.0.1](https://github.com/ixaxaar/pytorch-dnc/tree/v0.0.1) (2017-10-26)


\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*
