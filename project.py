import argparse
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

def main():
    num_nodes = 50
    num_rules = 100
    max_fn = 5
    iters = 100
    step_size = 0.3
    cost_dist = "gauss"
    pen_dist = "gauss"
    det_dist = "zipf"

    ruleset = generate_ruleset(num_nodes, num_rules, cost_dist, pen_dist, det_dist)
    hists = optimize(ruleset, num_nodes, num_rules, max_fn, iters, step_size, False)

    # Plot results generated
    lambdas = hists[0]
    s_nr_hist = hists[1]
    total_cost_hist = hists[2]
    total_fn_hist = hists[3]
    ruleset_hist = hists[4]
    plt.rcParams.update({'font.size': 16})

    # Plot primal variables changing for a randomly selected node
    node_chosen = np.random.choice(num_nodes)
    ax = plt.figure().add_subplot()
    for r in range(num_rules):
        s_nr = []
        for k in s_nr_hist:
            s_nr.append(k[node_chosen][r])
        ax.plot(s_nr)
    ax.set_title("Sampling rate progression for node " + str(node_chosen))
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("sampling rate")
    ax.set_xlim(0, iters)
    plt.show()

    # Plot dual variables chagning over time for all nodes
    ax = plt.figure().add_subplot()
    for n in range(num_nodes):
        lam = []
        for k in lambdas:
            lam.append(k[n])
        ax.plot(lam)
    ax.set_title("Lambda value progression")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("lambda value")
    ax.set_xlim(0, iters)
    plt.show()

    # Plot total cost changing over iterations
    ax = plt.figure().add_subplot()
    ax.plot(total_cost_hist)
    ax.set_title("Total cost for all nodes")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("Total cost")
    plt.show()

    # Plot total fn changing over iterations
    ax = plt.figure().add_subplot()
    ax.plot(total_fn_hist)
    # Plot the value of max_fn as a dotted line across
    max_val = np.full(len(total_fn_hist), max_fn / 100)
    ax.plot(max_val, "r--")
    ax.set_title("Total false negative rate for each node")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("Total false negative rate")
    ax.set_xlim(0, iters)
    plt.show()

    # Plot the sizes of the rulesets used within the network over iterations
    ax = plt.figure().add_subplot()
    for n in range(num_nodes):
        rs = []
        for k in ruleset_hist:
            rs.append(k[n])
        ax.plot(rs)
    ax.set_title("Ruleset size progression")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("Ruleset size")
    ax.set_xlim(0, iters)
    plt.show()

    hists = optimize(ruleset, num_nodes, num_rules, max_fn, iters, step_size, True)

    # Plot results generated
    lambdas = hists[0]
    s_nr_hist = hists[1]
    total_cost_hist = hists[2]
    total_fn_hist = hists[3]
    ruleset_hist = hists[4]
    plt.rcParams.update({'font.size': 16})

    # Plot primal variables changing for a randomly selected node
    node_chosen = np.random.choice(num_nodes)
    ax = plt.figure().add_subplot()
    for r in range(num_rules):
        s_nr = []
        for k in s_nr_hist:
            s_nr.append(k[node_chosen][r])
        ax.plot(s_nr)
    ax.set_title("Sampling rate progression for node " + str(node_chosen))
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("sampling rate")
    ax.set_xlim(0, iters)
    plt.show()

    # Plot dual variables chagning over time for all nodes
    ax = plt.figure().add_subplot()
    for n in range(num_nodes):
        lam = []
        for k in lambdas:
            lam.append(k[n])
        ax.plot(lam)
    ax.set_title("Lambda value progression")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("lambda value")
    ax.set_xlim(0, iters)
    plt.show()

    # Plot total cost changing over iterations
    ax = plt.figure().add_subplot()
    ax.plot(total_cost_hist)
    ax.set_title("Total cost for all nodes")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("Total cost")
    plt.show()

    # Plot total fn changing over iterations
    ax = plt.figure().add_subplot()
    ax.plot(total_fn_hist)
    # Plot the value of max_fn as a dotted line across
    max_val = np.full(len(total_fn_hist), max_fn / 100)
    ax.plot(max_val, "r--")
    ax.set_title("Total false negative rate for each node")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("Total false negative rate")
    ax.set_xlim(0, iters)
    plt.show()

    # Plot the sizes of the rulesets used within the network over iterations
    ax = plt.figure().add_subplot()
    for n in range(num_nodes):
        rs = []
        for k in ruleset_hist:
            rs.append(k[n])
        ax.plot(rs)
    ax.set_title("Ruleset size progression")
    ax.set_xlabel("iteration (k)")
    ax.set_ylabel("Ruleset size")
    ax.set_xlim(0, iters)
    plt.show()


# Perform gradient descent on the dual as specified in the paper
def optimize(ruleset, num_of_nodes, num_of_rules, max_fn, iters, step_size, pen=True):
    # Initialize the lambda values
    lam = np.full(num_of_nodes, 1.)
    lambdas = [copy.deepcopy(lam)]
    # Initialize the s_nr values to all by 1, i.e. starting with all rules enabled for all nodes in the network
    s_nr = [np.ones(num_of_rules) for i in range(num_of_nodes)]
    s_nr_hist = [copy.deepcopy(s_nr)]
    # Track the progression of the total cost to upkeep the system
    total_cost_hist = [sum(ruleset[0]) * num_of_nodes]
    # Track the progression of the total cumulative fn rate across every node
    total_fn_hist = [[sum([get_p_nr(ruleset, n, r) * (1 - s_nr[n][r]) for r in range(num_of_rules)]) / 100 for n in range(num_of_nodes)]]
    # Track the progression of the number of rules used by each node in the network
    ruleset_hist = [np.full(num_of_nodes, num_of_rules)]
    # Gradient descent on the dual
    for i in range(iters):
        # Perform optimization for each node
        for n in range(num_of_nodes):
            # Calculate gradient
            grad = sum([get_p_nr(ruleset, n, r) * (1 - s_nr[n][r]) for r in range(num_of_rules)]) - max_fn
            if pen:
                if grad > 0:
                    grad += 0.25
            # Update lambda for each node
            lam[n] += step_size * grad
            # Use calculated lambda value to update s_nr values
            for r in range(num_of_rules):
                s_nr[n][r] = 1 if lam[n] >= get_cost_r(ruleset, r) / (get_penalty_r(ruleset, r) * (get_p_nr(ruleset, n, r) ** 2)) else 0
        # Track the updated values
        lambdas.append(copy.deepcopy(lam))
        s_nr_hist.append(copy.deepcopy(s_nr))
        total_cost_hist.append(sum([get_cost_r(ruleset, r) * s_nr[n][r] for n in range(num_of_nodes) for r in range(num_of_rules)]))
        total_fn_hist.append([sum([get_p_nr(ruleset, n, r) * (1 - s_nr[n][r]) for r in range(num_of_rules)]) / 100 for n in range(num_of_nodes)])
        ruleset_hist.append([np.count_nonzero(s_nr[n]) for n in range(num_of_nodes)])

    hists = [lambdas, s_nr_hist, total_cost_hist, total_fn_hist, ruleset_hist]

    return hists


# Generates the costs, penalties, and detection probabilities of a set of rules, given the size of the network. The distributions available for selection are uniform, gaussian, lognormal, and zipfian.
def generate_ruleset(num_of_nodes, num_of_rules, cost_distribution="gauss", penalty_distribution="gauss", detection_distribution="zipf"):
    # Generate the value of cost_r
    match cost_distribution:
        case "uniform":
            costs = np.full(num_of_rules, 100)
        case "gauss":
            costs = np.random.normal(100, 20, num_of_rules)
        case "lognormal":
            costs = np.random.lognormal(50, 20, num_of_rules)
        case "zipf":
            costs = np.random.zipf(4, num_of_rules)
        case _:
            sys.exit("not a valid valid distribution type:" + cost_distribution)
    # Ensure no rules have negative cost
    for i in range(len(costs)):
        if costs[i] < 0:
            costs[i] = 1

    # Generate the value of penalty_r
    match penalty_distribution:
        case "uniform":
            pens = np.full(num_of_rules, 1000)
        case "gauss":
            pens = np.random.normal(1000, 100, num_of_rules)
        case "lognormal":
            pens = np.random.lognormal(500, 100, num_of_rules)
        case "zipf":
            pens = np.random.zipf(4, num_of_rules)
        case _:
            sys.exit("not a valid valid distribution type:" + penalty_distribution)
    # Ensure no rules incur negative penalties
    for i in range(len(pens)):
        if pens[i] < 0:
            pens[i] = 1
        
    # Generate the values of p_nr
    dets = []
    # Make sure each node in the network gets a unique set of p_nr values
    for i in range(num_of_nodes):
        match detection_distribution:
            case "uniform":
                dets.append(np.full(num_of_rules, 0.02))
            case "gauss":
                dets.append(np.random.normal(0.02, 0.055, num_of_rules))
            case "lognormal":
                dets.append(np.random.lognormal(0.01, 0.005, num_of_rules))
            case "zipf":
                det = np.random.zipf(4, num_of_rules)
                dets.append(np.divide(det, max(det) * 5))
            case _:
                sys.exit("not a valid valid distribution type:" + detection_distribution)

    # Combine all values generated into a ruleset
    ruleset = [costs, pens, dets]

    return ruleset
    
# Retrieves the cost associated with rule r from the ruleset provided
def get_cost_r(ruleset, r):
    return ruleset[0][r]

# Retrieves the penalty cost associated with rule r from the ruleset provided
def get_penalty_r(ruleset, r):
    return ruleset[1][r]

# Retrieves the detection probability of a given rule for a particular node from the ruleset provided
def get_p_nr(ruleset, n, r):
    return ruleset[2][n][r]

if __name__ == '__main__' : main()