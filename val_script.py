"""
Script for Validation portion of the cluster assignments
"""

def index_list(num, ind_list, ts_matrix):
    i = 0
    for z in zip(ind_list, ts_matrix):
        if z[0] == num and i == 0:
            output = np.array([z[1]])
            i += 1
        elif z[0] == num and i != 0:
            output = np.append(output, [z[1]], axis=0)
    return output

def main():
    louvain_ind = read_csv('mem.csv').values.T

    # TODO: generalize the ranges
    for f in files:
        ts_matrix = np.loadtxt('timeseries/' + f).T

        for i in range(1, 65):
            subject = louvain_ind[:722 * i][0]
            for j in range(4):
                i_list = index_list(j, subject, ts_matrix)
                avg = np.average(i_list, axis=1)
                Series(avg).to_csv("module_matrices/subject" + str(i)
                                    + "mod" + str(j), index=False, sep="\t")

if __name__ == "__main__":
    main()
