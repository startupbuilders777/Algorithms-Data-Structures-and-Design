#include "GeneralTree.h"
#include <vector>
#include <iostream>

using namespace std;

template <typename G>
void GeneralTree<G>::initialize()
{ //Initialize using  reads a pre-order traversal of a non-empty tree from standard input
    int numberOfChildren;
    int nodeNumber;
    cout << "hey";
    while (true)
    {
        if (!(cin >> nodeNumber >> numberOfChildren))
        {
            if (cin.eof())
                break;
            cin.clear();
            cin.ignore();
        }
        numOfNodes += numberOfChildren;
        std::cout << nodeNumber << endl;
        std::cout << numberOfChildren << endl;
    }
    //tree = new Node<G>(3);
}

template <typename G>
void GeneralTree<G>::initializeEachNode(Node *root, int children)
{
}


