#include <iostream>
#include <fstream>
#include <iomanip>
#include <array>
#include <string>
#include <map>

#include "json.hpp"

using namespace nlohmann;

json readDataFile(std::string filepath){
   std::ifstream ifs(filepath);
   json jf = json::parse(ifs);
   return jf;
}

void writeDataFile(std::string filepath, json data){    
   std::ofstream ofs(filepath);
   ofs << std::setw(4) << data << std::endl;
}

std::vector<std::vector<int>> getInitialTableau(std::vector<int> c,
       std::vector<std::vector<int>> A, std::vector<int> b)
{      
      for (size_t i = 0; i < A.size(); i++)
      {
         A[i].push_back(b[i]);         
      }
      
      c.push_back(0);
      A.push_back(c);

      return A;
}

bool canImprove(std::vector<std::vector<int>> tableau)
{
   auto lastRow = tableau[tableau.size()-1];
   bool retValue = false;

   for (auto &&i : lastRow)
   {
      if (i>0)
      {
         return true;
      }      
   }
   return false;
}
 
std::array<int,2> findPivotIndex(std::vector<std::vector<int>> tableau)
{
   //# pick minimum positive index of the last row
   auto lastRow = tableau[tableau.size()-1];
   int pivotColumn = -1;

   for (size_t i = 0; i < lastRow.size(); i++)
   {
      if (lastRow[i]>0)
      {
         pivotColumn = i;
         break;
      }      
   }
   
   // # check if unbounded
   bool unbounded = true;
   for (auto &&row : tableau)
   {
      if (row[pivotColumn] > 0)
      {
         unbounded = false;
      }      
   }

   if (unbounded)
   {
      std::cout<<"Linear program is unbounded."<<'\n';
      throw;
   }

   auto min = 0.0;
   int minIdx = -1;
   bool isDuplicateMin = false;

   // auto values = tableau.

   for (size_t i = 0; i < tableau.size()-2; i++) //the 'c' part is left out 
   {
      auto row = tableau[i];
      if (row[pivotColumn] > 0)
      {
         auto value = row[row.size()-1]/row[pivotColumn];
         // --
         // if (min == 0.0) //first iteration
         // {
         //    min = value;
         //    continue;
         // }

         if (min == 0.0 || value < min)
         {
            min = value;
            minIdx = i;
            isDuplicateMin = false;
         }
         else if (value == min)
         {
            isDuplicateMin = true;
         }         
         
      }
      
   }

   // # check for degeneracy: more than one minimizer of the quotient
   if (isDuplicateMin)
   {
      std::cout<< "The Linear program is degenerate."<<'\n';
      throw;
   }
   
   int pivotRow = minIdx;
   
   return {pivotRow, pivotColumn};
}

void pivotAbout(std::vector<std::vector<int>> tableau, std::array<int,2> pivots)
{
   int i = pivots[0];
   int j = pivots[1];

   auto pivotElement = tableau[i][j];
   
   // tableau[i] = [x / pivotDenom for x in tableau[i]]

   for (auto &&value : tableau[i])
   {
      value /= pivotElement;
   }
   
   for (size_t y = 0; y < tableau.size(); y++)
   {
      if (i!=y)
      {
         for (size_t x = 0; x < tableau[y].size(); x++)
         {
            tableau[y][x] -= (tableau[y][j] * tableau[i][x]);
         }
      }
   } 

}

std::map<int, double> getPrimeSolution(std::vector<std::vector<int>> tableau)
{
   std::map<int, double> results;

   for (size_t j = 0; j < tableau[0].size(); j++)
   {
      int idx = -1;
      for (size_t i = 0; i < tableau.size(); i++)
      {
         if (tableau[i][j] !=1 && tableau[i][j] !=0)
         {
            idx = -1;
            break;
         }

         if (tableau[i][j] ==1)
         {
            idx = (idx == -1) ? i : -1;
         }
      }
      results[j] = (idx == -1) ? 0 : tableau[idx][tableau.size()-1];
   }
   return results;   
}

double getObjectiveValue(std::vector<std::vector<int>> tableau)
{
   int i = tableau.size() -1;
   int j = tableau[0].size() -1;
   
   return tableau[i][j];
}

int main()
{
   std::vector<int>  c = {5, 4, 0, 0};
   std::vector<std::vector<int>> A = {{3, 5, 1, 0}, {4, 1, 0, 1}};
   std::vector<int> b = {78, 36};

   auto tableau = getInitialTableau(c, A, b);

   for (auto &&row : tableau)
   {
      for (auto &&i : row)
      {
         std::cout<<i<<' ';
      }
      std::cout<<'\n';     
   }

   int count = 0;
   while (canImprove(tableau))
   {
      std::cout<< "============   ITERATION "<< ++count <<"  ============   \n";
      auto pivotIndices = findPivotIndex(tableau);
      pivotAbout(tableau, pivotIndices);
      
   }
   
   auto results = getPrimeSolution(tableau);
   auto objValue = getObjectiveValue(tableau);
    
   // auto data = readDataFile("problem_data.json");
   std::cout<<"\n ==== yo !! ==== "<<'\n';
}

