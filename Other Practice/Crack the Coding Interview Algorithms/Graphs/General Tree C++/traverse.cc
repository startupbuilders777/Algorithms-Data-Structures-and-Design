
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include "GeneralTree.h"

using namespace std;

int main() {
    cout << "COOL" << endl;
  

    GeneralTree<int> * gt = new GeneralTree<int>();
    gt -> initialize();
   // gt -> print();
     delete gt;
    
    return 0;
}
