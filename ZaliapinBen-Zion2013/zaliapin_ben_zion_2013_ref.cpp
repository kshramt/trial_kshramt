// An implementation of Zaliapin and Ben-Zion (2013, JGR, http://doi.wiley.com/10.1002/jgrb.50179).
//
// # Note
// Distance log10(eta) can be -inf for earthquakes occurred at the same location.
//
// # License
// This program is distributed under the terms of the GNU General Public License version 3 (https://www.gnu.org/licenses/gpl-3.0.txt).

#include <GeographicLib/Geodesic.hpp>
#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const auto wgs84 = GeographicLib::Geodesic::WGS84();

template <typename M, typename B>
auto log_m_term(const M m, const B b) {
  return b * m;
}

template <typename T>
auto log_t_term(const T t) {
  return log10(t);
}

template <typename R, typename DF>
auto log_r_term(const R r, const DF df) {
  return df * log10(r);
}

template <typename T>
auto r_of(const T& lat1, const T& lon1, const T& depth1, const T& lat2, const T& lon2, const T& depth2) {
  T s12 = -999;
  wgs84.Inverse(lat1, lon1, lat2, lon2, s12);
  s12 /= 1000;                         // m -> km
  return hypot(s12, depth1 - depth2);  // This is only a crude approximation for a short distance.
}

int main(int argc, char* argv[]) {
  if (argc > 1) {
    cerr << "{\n"
         << "   echo $B $DF\n"
         << "   cat catalog.T_M_Lat°_Lon°_DepthKm\n"
         << "} | " << argv[0] << " > distance.j_i_logηij_logTij_logRij_logMi_and_more"
         << endl;
    exit(1);
  }
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.precision(std::numeric_limits<double>::max_digits10);
  cerr.precision(std::numeric_limits<double>::max_digits10);
  clog.precision(std::numeric_limits<double>::max_digits10);

  // read params
  double b, df;
  {
    string line;
    assert(getline(cin, line));
    istringstream iss(line);
    iss >> b >> df;
  }

  // read data
  vector<double> ts, ms, lats, lons, depths;
  {
    string line;
    for (; getline(cin, line);) {
      istringstream iss{line};
      double t, m, lat, lon, depth;
      iss >> t >> m >> lat >> lon >> depth;
      assert((-90 <= lat) && (lat <= 90));
      ts.push_back(t);
      ms.push_back(m);
      lats.push_back(lat);
      lons.push_back(lon);
      depths.push_back(depth);
    }
  }

  // output distances
  for (int j = ts.size() - 1; 0 < j; --j) {
    auto log_etaij_best = numeric_limits<double>::infinity();
    auto log_tij_best = numeric_limits<double>::infinity();
    auto log_rij_best = numeric_limits<double>::infinity();
    auto log_mi_best = -numeric_limits<double>::infinity();
    int i_best = -1;
    for (int i = 0; i < j; ++i) {
      if (ts[i] >= ts[j]) {
        break;
      }
      const auto log_tij = log_t_term(ts[j] - ts[i]);
      const auto log_mi = log_m_term(ms[i], b);
      const auto log_rij = log_r_term(r_of(lats[i], lons[i], depths[i], lats[j], lons[j], depths[j]), df);
      const auto log_etaij = log_tij + log_rij - log_mi;
      if (log_etaij < log_etaij_best) {
        log_etaij_best = log_etaij;
        log_tij_best = log_tij;
        log_rij_best = log_rij;
        log_mi_best = log_mi;
        i_best = i;
      }
    }
    cout << j
         << "\t" << i_best
         << "\t" << log_etaij_best
         << "\t" << log_tij_best
         << "\t" << log_rij_best
         << "\t" << log_mi_best
         << "\t" << ts[j]
         << "\t" << ms[j]
         << "\t" << lats[j]
         << "\t" << lons[j]
         << "\t" << depths[j]
         << "\t" << ts[i_best]
         << "\t" << ms[i_best]
         << "\t" << lats[i_best]
         << "\t" << lons[i_best]
         << "\t" << depths[i_best]
         << "\n";
  }

  return 0;
}
