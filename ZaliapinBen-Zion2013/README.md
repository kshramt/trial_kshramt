Implementation of the transformation method of Zaliapin and Ben-Zion (2013, JGR).

This code implements an O(N√N) algorithm and thus applicable to large catalogs.
For example, this code took about 6.5 s on a notebook computer to process a catalog, which contains 20,000 events.
On the same computer, it took 1121.6 s to process a catalog with 500,000 events.

# Usage

```bash
make
```

You may need to edit `-I` and `-L` for your environment.

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
