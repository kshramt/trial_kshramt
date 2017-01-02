An implementation of Zaliapin and Ben-Zion (2013, JGR).

This code implements an O(N^3/2^) algorithm and thus is applicable to large catalogs.
A benchmark result is as follows.

```
     N  Time (s)  Command
 20000       3.2  make N_EVENTS=20000
 40000       8.5  make N_EVENTS=40000
 80000      22.7  make N_EVENTS=80000
500000     334.1  make N_EVENTS=500000
```

# Usage

```bash
make all      # build ./zaliapin_ben_zion_2013.exe
make plot     # plot figures
make check    # run test cases
```

This code uses [GeographicLib](http://geographiclib.sourceforge.net/) to compute distances between events.
You may need to edit `-I` and `-L` in `Makefile` for your environment.

# References

```bib
@Article{ZaliapinBen-Zion2013,
  author    = {Zaliapin, Ilya and Ben-Zion, Yehuda},
  title     = {{Earthquake clusters in southern California I: Identification and stability}},
  journal   = {Journal of Geophysical Research: Solid Earth},
  year      = {2013},
  volume    = {118},
  number    = {6},
  pages     = {2847--2864},
  month     = {jun},
  doi       = {10.1002/jgrb.50179},
  issn      = {21699313},
  url       = {http://doi.wiley.com/10.1002/jgrb.50179},
}
```

# License

This program is distributed under the terms of [the GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.txt).
