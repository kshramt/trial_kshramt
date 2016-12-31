// Implement transformation method of Zaliapin and Ben-Zion (2013, JGR).
//
// # Note
// Distance log(eta) can be -inf for earthquakes occurred at the same location.

#include <cassert>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <GeographicLib/Geodesic.hpp>

using namespace std;

const auto wgs84 = GeographicLib::Geodesic::WGS84();

template<typename M, typename B>
auto log_m_term(const M m, const B b){
   return b*m*log(10);
}

template<typename T>
auto log_t_term(const T t){
   return log(t);
}

template<typename R, typename DF>
auto log_r_term(const R r, const DF df){
   return df*log(r);
}

template<typename T>
auto r_of(const T lat1, const T lon1, const T lat2, const T lon2){
   T s12 = -999;
   wgs84.Inverse(lat1, lon1, lat2, lon2, s12);
   return s12;
}

template<typename T>
auto sortperm(const vector<T>& xs){
   vector<size_t> inds(xs.size());
   iota(inds.begin(), inds.end(), 0);
   sort(inds.begin(), inds.end(),
        [&xs](auto i1, auto i2){return xs[i1] < xs[i2];});
   return inds;
}

template<typename T, typename I>
auto sort_by_inds(const vector<T>& xs, const vector<I>& inds){
   assert(xs.size() == inds.size());
   vector<T> ret(xs.size());
   for(size_t i = 0; i < inds.size(); i++){
      ret[i] = xs[inds[i]];
   }
   return ret;
}

template<typename T, typename I>
auto adaptive_split(const vector<T>& ts, const vector<T>& ms, const vector<T>& lats, const vector<T>& lons,
                    vector<vector<I>>& ibins, vector<vector<T>>& tbins, vector<vector<T>>& mbins, vector<vector<T>>& latbins, vector<vector<T>>& lonbins, // ibins[i_bin][i_i_bin] -> i_event
                    vector<I>& n_list, vector<T>& m_max_list, vector<T>& lat_min_list, vector<T>& lat_max_list, // xs[i_bin] -> x at i_bin
                    vector<I>& i_bin_list // i_bin_list[i_event] -> i_bin of i_event
                    ){
   const int n = ts.size();
   assert(ms.size() == n &&
          lats.size() == n &&
          lons.size() == n &&
          i_bin_list.size() == n
          );
   const int n_bins = ibins.size();
   assert(tbins.size() == n_bins &&
          mbins.size() == n_bins &&
          latbins.size() == n_bins &&
          lonbins.size() == n_bins &&
          m_max_list.size() == n_bins &&
          lat_min_list.size() == n_bins &&
          lat_max_list.size() == n_bins
          );

   const int n_per_bin = n/n_bins;
   int reminders = n - n_bins*n_per_bin;
   for(auto i_bin = 0; i_bin < n_bins; i_bin++){
      int plusone = 0;
      if(reminders > 0){
         plusone = 1;
         reminders -= 1;
      }
      n_list[i_bin] = n_per_bin + plusone;
   }

   const auto lat_inds = sortperm(lats);
   int i_lat = -1;
   for(int i_bin = 0; i_bin < n_bins; i_bin++){
      if(n_list[i_bin] > 0){
         for(int i_tmp = 0; i_tmp < n_list[i_bin]; i_tmp++){
            i_lat += 1;
            const auto i = lat_inds[i_lat];
            ibins[i_bin].push_back(i);
            tbins[i_bin].push_back(ts[i]);
            mbins[i_bin].push_back(ms[i]);
            latbins[i_bin].push_back(lats[i]);
            lonbins[i_bin].push_back(lons[i]);
            i_bin_list[i] = i_bin;
         }
         lat_min_list[i_bin] = latbins[i_bin][0];
         lat_max_list[i_bin] = latbins[i_bin][latbins[i_bin].size() - 1];
         m_max_list[i_bin] = *max_element(mbins[i_bin].begin(), mbins[i_bin].end());
         // sort vectors inside bin by t
         const auto t_inds = sortperm(tbins[i_bin]);
         ibins[i_bin] = sort_by_inds(ibins[i_bin], t_inds);
         tbins[i_bin] = sort_by_inds(tbins[i_bin], t_inds);
         mbins[i_bin] = sort_by_inds(mbins[i_bin], t_inds);
         latbins[i_bin] = sort_by_inds(latbins[i_bin], t_inds);
         lonbins[i_bin] = sort_by_inds(lonbins[i_bin], t_inds);
      }
   }
   assert(i_lat == n - 1);
}

// aggressively try early return
template<typename T, typename I>
auto find_parent(const vector<I>& is, const vector<T>& ts, const vector<T>& ms, const vector<T>& lats, const vector<T>& lons,
                 const T tj, const T latj, const T lonj,
                 const T m_max, const T dlat_min,
                 I& n,
                 I& i_best, T& log_etaij_best, T& log_tij_best, T& log_rij_best, T& log_mi_best,
                 const T b, const T df
                 ){
   const auto log_mi_possible_shortest = log_m_term(m_max, b);
   const auto log_rij_possible_shortest = log_r_term(r_of(latj + dlat_min, lonj, latj, lonj), df);
   T log_tij_pre = -numeric_limits<T>::infinity();
   for(int i = n - 1; i > -1; i--){
      if(ts[i] >= tj){
         n -= 1;
         continue;
      }
      const auto log_tij = log_t_term(tj - ts[i]);
      assert(log_tij >= log_tij_pre);
      log_tij_pre = log_tij;
      // quick return if possible
      if(log_tij + log_rij_possible_shortest - log_mi_possible_shortest >= log_etaij_best){
         break;
      }
      const auto log_mi = log_m_term(ms[i], b);
      assert(log_mi <= log_mi_possible_shortest);
      const auto log_rij = log_r_term(r_of(lats[i], lons[i], latj, lonj), df);
      assert(log_rij >= log_rij_possible_shortest);
      const auto log_etaij = log_tij + log_rij - log_mi;
      if(log_etaij < log_etaij_best){
         log_etaij_best = log_etaij;
         log_tij_best = log_tij;
         log_rij_best = log_rij;
         log_mi_best = log_mi;
         i_best = is[i];
      }
   }
}

int main(int argc, char* argv[]){
   if(argc > 1){
      cerr << "{\n"
           << "   echo $B $DF\n"
           << "   cat catalog.T_M_Lat°_Lon°\n"
           << "} | " << argv[0] << " > distance.j_i_logηij_logTij_logRij_logMi_and_more"
           << endl;
      exit(1);
   }
   cout.setf(ios_base::scientific, ios_base::floatfield);
   cout.precision(numeric_limits<double>::max_digits10);
   cerr.setf(ios_base::scientific, ios_base::floatfield);
   cerr.precision(numeric_limits<double>::max_digits10);

   // read params
   double b, df;
   {
      string line;
      assert(getline(cin, line));
      istringstream iss(line);
      iss >> b >> df;
   }

   // read data
   vector<double> ts, ms, lats, lons;
   string line;
   for(string line; getline(cin, line);){
      istringstream iss(line);
      double t, m, lat, lon;
      iss >> t >> m >> lat >> lon;
      assert((-90 <= lat) && (lat <= 90));
      ts.push_back(t);
      ms.push_back(m);
      lats.push_back(lat);
      lons.push_back(lon);
   }

   const int n = ts.size();
   const int n_bins = ceil(sqrt(n));
   vector<int> n_list(n_bins);
   vector<vector<int>> ibins(n_bins);
   vector<vector<double>> tbins(n_bins), mbins(n_bins), latbins(n_bins), lonbins(n_bins);
   vector<double> m_max_list(n_bins), lat_min_list(n_bins), lat_max_list(n_bins);
   vector<int> i_bin_list(n);

   adaptive_split(ts, ms, lats, lons,
                  ibins, tbins, mbins, latbins, lonbins,
                  n_list, m_max_list, lat_min_list, lat_max_list,
                  i_bin_list
                  );

   // output distances
   for(int j = ts.size() - 1; j > 0; j--){
      auto log_etaij_best = numeric_limits<double>::infinity();
      auto log_tij_best = numeric_limits<double>::infinity();
      auto log_rij_best = numeric_limits<double>::infinity();
      auto log_mi_best = -numeric_limits<double>::infinity();
      int i_best = -1;
      auto current_bin = i_bin_list[j];
      // I am expecting that the parent of j is inside current_bin
      find_parent(ibins[current_bin], tbins[current_bin], mbins[current_bin], latbins[current_bin], lonbins[current_bin],
                  ts[j], lats[j], lons[j],
                  m_max_list[current_bin], double(0),
                  n_list[current_bin],
                  i_best, log_etaij_best, log_tij_best, log_rij_best, log_mi_best,
                  b, df
                  );

      // search outward for efficient fast returns
      for(int delta_current_bin = 1; delta_current_bin < n_bins; delta_current_bin++){
         int direction = 1;
         for(int rep = 0; rep < 2; rep++){ // backward and forward
            direction *= -1;
            const auto i_bin = current_bin + direction*delta_current_bin;
            if(i_bin < 0 || n_bins <= i_bin){
               continue;
            }
            const auto minimum_delta_lat = direction > 0 ? lat_min_list[i_bin] - lats[j] : lat_max_list[i_bin] - lats[j];
            find_parent(ibins[i_bin], tbins[i_bin], mbins[i_bin], latbins[i_bin], lonbins[i_bin],
                        ts[j], lats[j], lons[j],
                        m_max_list[i_bin], minimum_delta_lat,
                        n_list[i_bin],
                        i_best, log_etaij_best, log_tij_best, log_rij_best, log_mi_best,
                        b, df
                        );
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
           << "\t" << ts[i_best]
           << "\t" << ms[i_best]
           << "\t" << lats[i_best]
           << "\t" << lons[i_best]
           << "\n";
   }

   return 0;
}
