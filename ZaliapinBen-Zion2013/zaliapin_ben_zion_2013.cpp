// Implement transformation method of Zaliapin and Ben-Zion (2016, JGR).
//
// # Usage
//     g++ -O3 -march=native -Wa,-q -o zaliapin_ben_zion_2013.exe zaliapin_ben_zion_2013.cpp -L/opt/local/lib -lGeographic
//     { echo 1 1.6 ; cat catalog.tmyx ; } | time ./zaliapin_ben_zion_2013.exe | tee catalog.dists
//
// # Note
// Distance eta can be -inf for earthquakes occurred at the same location.

#include <iostream>
#include <limits>
#include <tuple>
#include <sstream>
#include <string>
#include <vector>
#include <GeographicLib/Geodesic.hpp>

using namespace std;

const auto wgs84 = GeographicLib::Geodesic::WGS84();

template<typename M, typename B>
auto log_m_term(M m, B b){
   return b*m*log(10);
}

template<typename T>
auto log_t_term(T t){
   return log(t);
}

template<typename R, typename DF>
auto log_r_term(R r, DF df){
   return df*log(r);
}

template<typename T>
auto r_of(T lat1, T lon1, T lat2, T lon2){
   auto s12 = -999.0;
   wgs84.Inverse(lat1, lon1, lat2, lon2, s12);
   return s12;
}

int main(int argc, char* argv[]){

   // read params
   double b, df;
   cin >> b >> df;

   // read data
   vector<double> ts, ms, lats, lons;
   string line;
   for(string line; getline(cin, line);){
      istringstream iss(line);
      double t, m, lat, lon;
      iss >> t >> m >> lat >> lon;
      ts.push_back(t);
      ms.push_back(m);
      lats.push_back(lat);
      lons.push_back(lon);
   }

   // output distances
   cout.setf(ios_base::scientific, ios_base::floatfield);
   cout.precision(numeric_limits<double>::max_digits10);
   for(int j = 1; j < ts.size(); j++){
      auto log_etaij_best = numeric_limits<double>::infinity();
      auto log_tij_best = numeric_limits<double>::infinity();
      auto log_rij_best = numeric_limits<double>::infinity();
      auto log_mi_best = numeric_limits<double>::infinity();
      auto i_best = -1;
      for(int i = 0; i < j; i++){
         auto log_mi = log_m_term(ms[i], b);
         if(ts[i] >= ts[j]){
            break;
         }
         auto log_tij = log_t_term(ts[j] - ts[i]);
         auto log_rij = log_r_term(r_of(lats[i], lons[i], lats[j], lons[j]), df);
         auto log_etaij = log_tij + log_rij - log_mi;
         if(log_etaij < log_etaij_best){
            log_etaij_best = log_etaij;
            log_tij_best = log_tij;
            log_rij_best = log_rij;
            log_mi_best = log_mi;
            i_best = i;
         }
      }
      cout << j << "\t"
           << i_best << "\t"
           << log_etaij_best << "\t"
           << log_tij_best << "\t"
           << log_rij_best << "\t"
           << log_mi_best
           << endl;
   }

   return 0;
}
