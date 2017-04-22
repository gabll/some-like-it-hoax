
import numpy as np
import random
import unittest


class Item(object):
    """Class representing an item.  See VotingGraph for general comments."""

    def __init__(self, id, true_value=None, is_testing=False, inertia=1.0):
        """
        Initializes an item.
        :param known_value: None if we don't know the truth value of the item; +1 for True, -1 for False.
        :param true_value: The true value, if known.
        :param is_testing: If true, then we don't use the true value in the inference.
        """
        self.id = id
        self.inferred_value = None # Value computed by the inference.
        # True value (ground truth)
        self.true_value = None if true_value is None else 1.0 if true_value else -1.0
        self.is_testing = is_testing
        # Inertia for changing the belief.
        self.inertia = inertia
        self.users = [] # List of users who voted on the item.

    def __repr__(self):
        return repr(dict(
            id=self.id,
            inferred_value=self.inferred_value,
            true_value=self.true_value,
            is_testing=self.is_testing,
            correct=self.is_correctly_classified(),
            users=[u.id for _, u in self.users]
        ))

    def add_user(self, u, pol):
        """Add user u with polarity pol to the item."""
        self.users.append((pol, u))

    def set_is_testing(self, is_testing):
        self.is_testing = is_testing

    def set_true_value(self, true_value):
        self.true_value = None if true_value is None else 1.0 if true_value else -1.0

    def is_correctly_classified(self):
        """Returns (t, c), where t is 1 whenever we can measure whether the
        classification is correct, and c is the correctness (0 = wrong, 1 = correct).
        """
        if (not self.is_testing) or self.true_value is None or self.inferred_value is None:
            return (0.0, 0.0)
        else:
            return (1.0, 1.0) if self.inferred_value * self.true_value > 0 else (1.0, 0.0)

    def initialize(self):
        """Initializes the item, setting its inferred_value to the known value
        unless we are testing."""
        self.inferred_value = None if self.is_testing else self.true_value

    def compute_item_value(self):
        """Performs one step of inference for the item."""
        if self.true_value is None or self.is_testing:
            # Performs actual inference
            pos_w = neg_w = self.inertia
            for pol, u in self.users:
                if u.inferred_value is not None:
                    delta = pol * u.inferred_value
                    # print "  Item ", self.id, "from user", u.id, "polarity", pol, "delta:", delta # debug
                    if delta > 0:
                        pos_w += delta
                    else:
                        neg_w -= delta
            self.inferred_value = (pos_w - neg_w) / (pos_w + neg_w)
            # print "Item", self.id, "inferred value", pos_w, neg_w, self.inferred_value # debug
        else:
            # The value is known, and we are allowed to use it.
            self.inferred_value = self.true_value
            # print "Item", self.id, "inferred = truth", self.inferred_value

    def degree(self):
        return len(self.users)


class User(object):
    """Class representing a user.  See VotingGraph for general comments."""

    def __init__(self, id, known_value=None, neg_weight=1.0, pos_weight=1.5):
        """Initializes a user.
        :param known_value: None if we don't know the goodness of the user, otherwise, the goodness of the
        user as a number between 0 and 1.
        :param pos_weight: Initial positive weight of a user.
        :param neg_weight: Initial (offset) negative weight of a user.  These two weights correspond to how many
            correct and wrong likes we have seen the user do in the past, and is used to initialize the algorithm
            so we need automatically some evidence before we believe a user is right or wrong, with a weak
            initial positive bias.
        """
        self.id = id
        self.initial_pos_weight = pos_weight
        self.initial_neg_weight = neg_weight
        self.known_value = known_value
        self.inferred_value = known_value
        self.items = []

    def __repr__(self):
        return repr(dict(
            id=self.id,
            known_value=self.known_value,
            inferred_value=self.inferred_value,
            items=[it.id for _, it in self.items]
        ))

    def add_item(self, it, pol):
        """ Adds an item it with polarity pol to the user. """
        self.items.append((pol, it))

    def initialize(self):
        self.inferred_value = None

    def compute_user_value(self):
        """Performs one step of inference on the user."""
        pos_w = float(self.initial_pos_weight)
        neg_w = float(self.initial_neg_weight)
        # Iterates over the items.
        for pol, it in self.items:
            if it.inferred_value is not None:
                delta = pol * it.inferred_value
                # print "  User", self.id, "from item", it.id, "polarity", pol, "delta:", delta # debug
                if delta > 0:
                    pos_w += delta
                else:
                    neg_w -= delta
        self.inferred_value = (pos_w - neg_w) / (pos_w + neg_w)
        # print "User", self.id, "inferred value:", pos_w, neg_w, self.inferred_value


class VotingGraph(object):
    """This class represents a bipartite graph of users and items.  Users and items are connected via
    edges that can be labeled either +1 (True) or -1 (False) according to whether the user thinks
    the item is true or false.  We could use booleans as edge labels, but we would lose the ability to
    represent levels of certainty later on, so for now, we use integers.
    It is possible to label items as +1 (True) or -1 (False), in which case the items have a known truth
    value; alternatively, the item label can be left to None, and it can be later inferred."""

    def __init__(self, start_pos_weight=5.01, start_neg_weight=5.0, item_inertia=5.0):
        """Initializes the graph.
        The user initially has a value of
        (start_pos_weight - start_neg_weight) / (start_pos_weight + start_neg_weight),
        and start_pos_weight and start_neg_weight essentially give the inertia with which
        we modify the opinion about a user as new evidence accrues.
        :param start_pos_weight: Initial positive weight on a user.
        :param start_neg_weight: Initial negative weight on a user.
        :param item_inertia: Inertia on an item.
        """
        # Dictionary from items to users.
        self.items = {} # Dictionary from id to item
        self.users = {} # Dictionary from id to user
        self.edges = [] # To sample.
        self.start_pos_weight = start_pos_weight
        self.start_neg_weight = start_neg_weight
        self.item_inertia = item_inertia


    def add_edge(self, user_id, item_id, pol=1, item_true_value=None):
        """Adds an edge to the graph, from a user to an item.
        :param user_id: id of the user.
        :param item_id: id of the item.
        :param pol: polarity of the edge: +1 if the user thinks the item is reputable, -1 otherwise.
        :param item_true_value: true value of the item, if known for training.
        """
        # Creates user if necessary.
        u = self.users.get(user_id)
        if u is None:
            u = User(user_id, pos_weight=self.start_pos_weight, neg_weight=self.start_neg_weight)
            self.users[user_id] = u
        # Create item if necessary.
        it = self.items.get(item_id)
        if it is None:
            it = Item(item_id, true_value=item_true_value, inertia=self.item_inertia)
            self.items[item_id] = it
        # Adds edge.
        u.add_item(it, pol)
        it.add_user(u, pol)
        self.edges.append((item_id, user_id))


    def get_user_ids(self):
        return self.users.keys()

    def get_item_ids(self):
        return self.items.keys()

    def iter_items(self):
        return self.items.values()

    def iter_users(self):
        return self.users.values()


    def get_user(self, user_id):
        return self.users.get(user_id)

    def get_item(self, item_id):
        return self.items.get(item_id)

    def perform_inference(self, num_iterations=5):
        """Performs inference on the graph."""
        for u in self.users.values():
            u.initialize()
        for it in self.items.values():
            it.initialize()
        for _ in range(num_iterations):
            [u.compute_user_value() for u in self.users.values()]
            [it.compute_item_value() for it in self.items.values()]

    def print_stats(self):
        """Prints graph statistics, mainly for testing purposes"""
        num_items_truth_known = len([it for it in self.iter_items() if it.true_value is not None])
        num_items_inferred_known = len([it for it in self.iter_items() if it.inferred_value is not None])
        num_items_testing = len([it for it in self.iter_items() if it.is_testing])
        print "Num items:", len(self.items)
        print "Num items with truth known:", num_items_truth_known
        print "Num items with inferred known:", num_items_inferred_known
        print "Num items testing:", num_items_testing
        print "Min degree:", min([it.degree() for it in self.iter_items()])
        print "Num users:", len(self.users)
        print "Num likes:", sum([len(u.items) for u in self.users.values()])

    def evaluate_inference(self, fraction=100, num_runs=50):
        """
        Evaluation function we use.
        :param num_chunks: In how many chunks we split the data.
        :param num_eval_chunks: number of chunks used for evaluation, if different from num_chunks.
        :param learn_from_most: If True, we learn from all but one chunk (on which we test).
                If False, we learn from one chunk, test on all others.
        :return: A dictionary of values for creating plots and displaying performance.
        """
        item_ids = self.get_item_ids()
        num_items = len(item_ids)
        chunk_len = num_items / fraction
        correct = 0.0
        tot = 0.0
        ratios = []
        # We split the items into k portions, and we cycle, considering each
        # of these portions the testing items.
        for run_idx in range(num_runs):
            random.shuffle(item_ids)
            for i, it_id in enumerate(item_ids):
                self.get_item(it_id).set_is_testing(i >= chunk_len)
            # Performs the inference.
            self.perform_inference()
            # vg.print_stats()
            # print("Performed inference for chunk %d" % chunk_idx)
            # Measures the accuracy.
            run_correct = 0.0
            run_total = 0.0
            for it_id in item_ids[chunk_len:]:
                it = self.get_item(it_id)
                tot_cor = it.is_correctly_classified()
                tot += tot_cor[0]
                correct += tot_cor[1]
                run_total += tot_cor[0]
                run_correct += tot_cor[1]
            run_ratio = run_correct / run_total
            ratios.append(run_ratio)
            print "One run result:", run_ratio
        # Computes the averages.
        ratio_correct = correct / tot
        return dict(
            ratio_correct=ratio_correct,
            stdev=np.std(ratios),
        )


    def evaluate_inference_given_learning(self, fun_learning):
        """
        :param: A function fun_learning, which tells us if we have to learn from an item
            with a given id yes or no.
        :return: The ratio of correctly classified items.
        """
        item_ids = self.get_item_ids()
        # We want to measure the accuracy for posts that have at least 1, 2, ..., LIKES_MEASURED likes.
        correct = 0.0
        tot = 0.0
        # Sets which items are learning, and which are testing.
        test_items = []
        for it_id in item_ids:
            is_testing = not fun_learning(it_id)
            self.get_item(it_id).set_is_testing(is_testing)
            if is_testing:
                test_items.append(it_id)
        # Performs the inference.
        self.perform_inference()
        # vg.print_stats()
        # print("Performed inference for chunk %d" % chunk_idx)
        # Measures the accuracy.
        for it_id in test_items:
            it = self.get_item(it_id)
            tot_cor = it.is_correctly_classified()
            tot += tot_cor[0]
            correct += tot_cor[1]
        return correct / tot if tot > 0 else 1


    def evaluate_inference_selecting_prop_likes(self, frac=0.1):
        """
        Evaluates the accuracy over ONE run, selecting a fraction frac of items, where each item
        is selected with probability proportional to the number of links.
        :param frac: Fraction of items considered.
        :return: The ratio of correct items. 
        """
        learn_items = set()
        item_ids = self.get_item_ids()
        # How many items do we have to pick?
        num_items = max(1, int(0.5 + frac * len(item_ids)))
        # How many we have picked already?
        num_picked = 0
        while num_picked < num_items:
            it_id, _ = random.choice(self.edges)
            if it_id not in learn_items:
                num_picked += 1
                learn_items.add(it_id)
        # Ok, now we do the learning.
        for it_id in item_ids:
            self.get_item(it_id).set_is_testing(it_id not in learn_items)
        self.perform_inference()
        correct = 0.0
        tot = 0.0
        for it_id in item_ids:
            it = self.get_item(it_id)
            tot_cor = it.is_correctly_classified()
            tot += tot_cor[0]
            correct += tot_cor[1]
        return correct / tot if tot > 0 else 1.0


class TestGraph(unittest.TestCase):

    def print_graph(self, vg, s):
        print ("---- " + s + " ----")
        print("Items:")
        for it in vg.iter_items():
            print(it)
        print("Users:")
        for u in vg.iter_users():
            print(u)

    def test_simple(self):
        """One item only, everybody tells the truth."""
        vg = VotingGraph()
        vg.add_edge("u1", "i1", item_true_value=True)
        vg.add_edge("u2", "i1", item_true_value=True)
        vg.get_item("i1").set_is_testing(True)
        vg.perform_inference()
        # self.print_graph(vg, 'simple')
        self.is_correct(vg)

    def is_correct(self, vg):
        cl = [it.is_correctly_classified() for it in vg.iter_items()]
        self.assertEqual(sum([c[0] for c in cl]), sum([c[1] for c in cl]))

    def test_true_and_false(self):
        vg = VotingGraph()
        vg.add_edge("u1", "i1")
        vg.add_edge("u2", "i1")
        vg.add_edge("u1", "i2")
        vg.add_edge("u2", "i2")
        vg.add_edge("u3", "i3")
        vg.add_edge("u3", "i4")
        vg.add_edge("u4", "i3")
        vg.add_edge("u4", "i4")
        i1 = vg.get_item('i1')
        i2 = vg.get_item('i2')
        i3 = vg.get_item('i3')
        i4 = vg.get_item('i4')
        i1.set_true_value(True)
        i2.set_true_value(True)
        i3.set_true_value(False)
        i4.set_true_value(False)
        i2.set_is_testing(True)
        i4.set_is_testing(True)
        vg.perform_inference()
        self.print_graph(vg, 'true and false final')
        self.is_correct(vg)
        self.assertEqual(i1.degree(), 2)
        vg.print_stats()


    def test_false_users(self):
        vg = VotingGraph()
        vg.add_edge("u3", "i3", item_true_value=False)
        vg.add_edge("u3", "i4", item_true_value=False)
        vg.add_edge("u4", "i3")
        vg.add_edge("u4", "i4")
        vg.get_item("i4").set_is_testing(True)
        vg.perform_inference()
        # self.print_graph(vg, 'false users final')
        self.is_correct(vg)