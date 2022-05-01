#include <bits/stdc++.h>
#define N 100050
using namespace std;

const double eps = 1e-5;

int num_of_walks = 40;
int seq_len = 4;
int o[N];
char dis[N][N];
bool vis[N];
int n, m;
vector<int> E[N];
vector<double> p_merw[N];

class AliasTable{
    public:
        vector<int> A, B;
        vector<double> S;
    public:
        AliasTable () {}
        void init(vector<int> &a, vector<double> &p) {
            queue<int> qA, qB;
            queue<double> pA, pB;
            int n = (int)a.size();
            for (int i=0;i<n;i++) p[i] = p[i] * n;
            for (int i=0;i<n;i++)
                if (p[i] > 1.0) {
                    qA.push(a[i]);
                    pA.push(p[i]);  
                } else {
                    qB.push(a[i]);
                    pB.push(p[i]);
                }
            while (!qA.empty() && !qB.empty()) {
                int idA = qA.front(); qA.pop();
                double probA = pA.front(); pA.pop();
                int idB = qB.front(); qB.pop();
                double probB = pB.front(); pB.pop();
                
                A.push_back(idA);
                B.push_back(idB);
                S.push_back(probB);

                double res = probA-(1.0-probB);

                if (abs(res-1.0) < eps) {
                    A.push_back(idA);
                    B.push_back(idA);
                    S.push_back(res);
                    continue;
                }

                if (res > 1.0) {
                    qA.push(idA);
                    pA.push(res);
                } else {
                    qB.push(idA);
                    pB.push(res);
                }
            }

            while (!qA.empty()) {
                int idA = qA.front(); qA.pop();
                pA.pop();
                A.push_back(idA);
                B.push_back(idA);
                S.push_back(1.0);
            }

            while (!qB.empty()) {
                int idB = qB.front(); qB.pop();
                pB.pop();
                A.push_back(idB);
                B.push_back(idB);
                S.push_back(1.0);
            }
        }

        int roll() {


	    if ((int)A.size() == 0) {
		cerr << "ERROR:: A.size() == 0 in Alias Table" << endl;
		exit(0);
	    }
            int x = rand() % ((int)A.size());
            double p = 1.0 * rand() / RAND_MAX;
            return p>S[x] ? A[x] : B[x];
        }

}AT[N];


void link(int u, int v, double p)
{
    E[u].push_back(v);
    p_merw[u].push_back(p);
}

void bfs(int S)
{
    queue<int> q;
    q.push(S);
    dis[S][S] = 1;
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        if (dis[S][u] > seq_len)
            return;
        for (int i = 0; i < (int)E[u].size(); i++)
        {
            int v = E[u][i];
            if (dis[S][v] == 0)
            {
                dis[S][v] = dis[S][u] + 1;
                q.push(v);
            }
        }
    }
    return;
}

int main(int argc, char *argv[])
{

    if (argc != 4)
    {
        cerr << "ERROR: Incorrect number of parameters. " << endl;
        return 0;
    }

    stringstream ss1, ss2;
    ss1.str("");
    ss1 << "./edge_input/";
    ss1 << argv[1];
    ss1 << ".in";

    num_of_walks = atoi(argv[2]);
    seq_len = atoi(argv[3]);

    freopen(ss1.str().c_str(), "r", stdin);
    srand(time(0));
    scanf("%d%d", &n, &m);

    cerr << argv[1] << ": " << n << endl;

    for (int i = 1; i <= m; i++)
    {
        int u, v;
        double p;   
        scanf("%d%d%lf", &u, &v, &p);
        link(u, v, p);
    }

    for (int i=0;i<n;i++) {
        AT[i].init(E[i], p_merw[i]);
    }

    for (int i = 0; i < n; i++)
        bfs(i);

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        ss2.str("");
        // ss2 << "./preprocess/";
        ss2 << "/data/syf/rw/";
        ss2 << argv[1];
        ss2 << "_";
        ss2 << argv[2];
        ss2 << "_";
        ss2 << argv[3];
        ss2 << "_";
        ss2 << epoch;
        ss2 << "_merw";
	ss2 << ".txt";
        freopen(ss2.str().c_str(), "w", stdout);
        for (int st = 0; st < n; st++)
        {
            for (int i = 0; i < num_of_walks; i++)
            {
                int u = st;
                printf("[");
                for (int _ = 0; _ < seq_len; _++)
                {
                    printf("%d", u);
                    o[_] = dis[st][u];
                    printf(", ");
                    int tot = E[u].size();
                    int g = AT[u].roll();
                    u = g;
                }

                for (int _ = 0; _ < seq_len; _++)
                {
                    printf("%d", o[_] - 1);
                    if (_ != seq_len - 1)
                        printf(", ");
                }

                printf("]\n");
            }
        }
        fclose(stdout);
    }
    fclose(stdin);
    return 0;
}
