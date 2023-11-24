import itertools

import pulp
from pulp import PULP_CBC_CMD, GUROBI, GUROBI_CMD


from instance_48 import DISTANCE


def solve_tsp(distances: list, num_city):
    m = pulp.LpProblem('tsp', pulp.LpMinimize)
    big_m = num_city
    city_list = list(range(num_city))
    arc_list = list(itertools.product(city_list, repeat=2))

    x = pulp.LpVariable.dicts('x', arc_list, cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts('u', city_list, cat=pulp.LpContinuous, lowBound=0)

    m += pulp.lpSum(distances[i][j] * x[i, j] for i in city_list for j in city_list)

    for i in city_list:
        m += (pulp.lpSum(x[i, j] for j in city_list) == 1, f'outflow_{i}')
        m += (pulp.lpSum(x[j, i] for j in city_list) == 1, f'inflow_{i}')
        m += (x[i, i] == 0, f'redundant_{i}')

    for i, j in arc_list:
        if i == j or j == 0:
            continue
        m += (u[j] >= u[i] + 1 + big_m * (x[i, j] - 1), f'mtz_{i}_{j}')

    time_limit_in_seconds = 3 * 60 * 60
    m.writeLP('test.lp')
    m.solve(PULP_CBC_CMD(timeLimit=time_limit_in_seconds, gapRel=0))

    sol = list()
    i = 0
    while True:
        for j in city_list:
            if x[i, j].value() > 0.9:
                sol.append((i, j))
                i = j
                break
        if i == 0:
            break
    return sol, pulp.value(m.objective)


def main():
    distance_list = DISTANCE
    num_city = 48
    sol, objective = solve_tsp(distance_list, num_city)
    print(sol)
    print(objective)
    return


if __name__ == '__main__':
    main()
