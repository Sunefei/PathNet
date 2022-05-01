#include <bits/stdc++.h>
#define N 1000500
#define walk_len 3
using namespace std;
int n, m;
int ans[N][16], feature[N];
vector<int> E[N];

void dfs(int u, int f, int k, int mask) {
    if (k == 0) {
        ans[f][mask]++;
        return ;
    }
    for (int i=0;i<(int)E[u].size();i++) {
        int v = E[u][i];
        dfs(v, f, k-1, ((mask<<1)|feature[v]));
    }
}

int main() {
    freopen("../preprocess/citeseer.in","r",stdin);
    freopen("citeseer_label.out", "w", stdout);
    scanf("%d%d", &n, &m);
    for (int i=0;i<n;i++) feature[i] = i&1;
    for (int i=1;i<=m;i++) {
        int u, v;
        scanf("%d%d", &u, &v);
        E[u].push_back(v);
        E[v].push_back(u);
    }

    for (int i=0;i<n;i++) {
        dfs(i, i, walk_len, 0);
    }

    for (int i=0;i<n;i++) {
        int o = 0;
        for (int j=0;j<16;j++) {
            if (ans[i][j] > ans[i][o]) o = j;
        }
        //printf("%d%c", o, i==n-1?'\n':' ');
        printf("%d\n", o);
    }
    fclose(stdin);
    fclose(stdout);
    return 0;
}
