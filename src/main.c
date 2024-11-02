#include "../include/graph.h"

int main(){
    Graph * undirected = createGraph(3, false);
    undirected->addEdge(undirected, 0,1);
    undirected->addEdge(undirected, 0, 2);

    undirected->print(undirected);

}
