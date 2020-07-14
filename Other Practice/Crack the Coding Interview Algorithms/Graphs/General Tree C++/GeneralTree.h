#ifndef TREE_H
#define TREE_H

#include <string>
#include <vector>
#include <iostream>

template <typename G>
class GeneralTree
{
    class Node
    {
      public:
        Node(): numberOfChildren(0) {}
        Node(int numberOfChildren) : numberOfChildren(numberOfChildren), children(std::vector<Node *>(numberOfChildren)) {}
        Node(int numberOfChildren, G *data) : data(data), numberOfChildren(numberOfChildren) {}

        ~Node()
        {
            delete this->data;
            for (typename std::vector<Node *>::iterator it = this->children.begin(); it != this->children.end(); ++it)
            {
                delete *it;
            }
        }

      private:
        int numberOfChildren;
        std::vector<Node *> children;
        G * data;
    };

  public:
    GeneralTree()
    {
        numOfNodes = 1;
        tree = new Node();
    }

    virtual ~GeneralTree()
    {
        delete this->tree;
    }

    void initialize(); //Initialize using  reads a pre-order traversal of a non-empty tree from standard input
    void print();      //Prints the corresponding post-order traversal for that tree

    int getNumberOfNodes() const
    {
        return numOfNodes;
    }

  private:
    void initializeEachNode(Node *node, int children);

    Node *tree;
    int numOfNodes;
};

#endif