#include<iostream>
#include<unordered_set>
using namespace std;
int main(int argc, char const *argv[])
{
    unordered_set<int> v  = {1,100,10,70,100};
    for (auto &a:v){
        cout<<a<<endl;
    }
    return 0;
}
