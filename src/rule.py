import gmpy2
from gmpy2 import mpz
import re
import numpy as np
import sys

labels = None
lead_one = None

"""
	Python implementation of a rule

	name: string representation of the rule
	tt: mpz representation of the samples that the rule captures
	cardinality: amount of samples in a rule
	captured: number of samples the rule captures
	predict: what outcome the rule predicts (frequentist)
	correct: percentage of samples that the rule predicts correctly

"""


class rule(object):

    def __init__(self, name, num, truthtable):
        self.num = num
        self.name = name
        self.tt = truthtable
        # subtract 1 to remove leading 1
        self.cardinality = truthtable.num_digits(2) - 1
        self.captured = gmpy2.popcount(truthtable) - 1
        # if more than half the samples are captured
        if labels and self.captured > 0:
            self.corr = gmpy2.popcount(self.tt & labels[1].tt) - 1
            if self.corr >= (self.captured / 2):
                self.predict = 1
                self.correct = self.corr / float(self.captured)
            else:
                self.predict = 0
                self.correct = (self.captured - self.corr) / \
                    float(self.captured)
        else:
            self.predict = 1
            self.correct = 0
            self.corr = 0

"""
	Python implementation of a ruleset

	nrules: number of rules
	rules: list of rules
	tt: mpz representation of samples this ruleset captures
"""


class ruleset(object):

    def __init__(self, nrules, rules):
        self.nrules = nrules
        self.rules = rules
        self.tt = mpz(pow(2, rules[0].cardinality))
        for r in rules:
            self.tt = rule_vor(self.tt, r.tt)

"""
	Python implementation of make_default

	Returns a mpz object consisting of length ones

	Note: in order to ensure you have a leading one, pass in
	a length that is 1 greater than your number of samples
"""


def make_all_ones(length):
    ones = pow(2, length) - 1
    default_tt = mpz(ones)
    return default_tt

"""
	Python implementation of make_zero

	Returns a list of mpz object of 0
"""

def make_zeros(length):
    return [mpz(0)]*length

"""
	Returns a rule that captures everything not captured in the prefix
"""


def make_default(length, prefix):
    default_tt = make_all_ones(length)
    for r in prefix:
        default_tt = rule_vandnot(default_tt, r.tt)[0]
    default_rule = rule('default', default_tt)
    return default_rule

"""
	Reads in the file and creates a ruleset out of it
"""


def parse(filename):
    data = []
    with open(filename) as f:
        for line in f:
            parsed = line.split(None, 1)
            rulename = parsed[0]
            truthtable = re.sub('[\s+]', '', parsed[1])
            # to prevent clearing of leading zeroes
            truthtable = '1' + truthtable
            bitstring = mpz(truthtable, 2)
            r = rule(rulename, bitstring)
            data.append(r)
    return data

"""
	Python implementation of rules_init
"""


def rules_init(xs, ys):
    global labels, lead_one
    labels = parse(ys)
    x = parse(xs)
    lead_one = mpz(pow(2, x[0].cardinality))
    return x

"""
	Python implementation of rule_copy
"""


def rule_copy(r):
    return copy.deepcopy(r)


"""
    Python implementation of rule_vectompz

    Convert a binary vector to a mpz object

    Note: in order to ensure you have a leading one,
    add '1' in the front
"""

def rule_vectompz(vec):
    return mpz('1' + "".join([str(i) for i in vec]), 2)

"""
    Python implementation of rule_mpztovec

    Convert a mpz object to a binary vector
"""


def rule_mpztovec(tt):
    # remove the leading one
    vec = list(tt.digits(2)[1:])
    return list(map(int, vec))


"""
    Python implementation of rule_vand

    Takes in two truthtables
    Returns the and of the truthtables
    as well as the number of ones in the and
"""


def rule_vand(tt1, tt2):
    vand = tt1 & tt2
    # subtract 1 to remove leading ones
    cnt = gmpy2.popcount(vand) - 1
    return vand, cnt

"""
	Python implementation of rule_vor

	Takes in two truthtables
	Returns the 'or' of the truthtables
	as well as the number of ones in the 'or'
"""


def rule_vor(tt1, tt2):
    vor = tt1 | tt2
    # subtract 1 to remove leading ones
    cnt = gmpy2.popcount(vor) - 1
    return vor, cnt


"""
	Python implementation of rule_vxor

	Takes in two truthtables
	Returns the number of ones in the 'xor'
"""

def rule_vxor(tt1, tt2):
    vxor = tt1 ^ tt2
    # subtract 1 to remove leading ones
    cnt = gmpy2.popcount(vxor)
    return cnt 


"""
	Python implementation of rule_vandnot

	Takes in two truthtables
	Returns the 'and' of tt1 with the 'not' of tt2
	as well as the number of ones in the 'and'
"""


def rule_vandnot(tt1, tt2):
    # or in lead_one so we don't lose our leading digits
    vand = tt1 & (~tt2 | lead_one)
    # subtract 1 to remove leading ones
    cnt = gmpy2.popcount(vand) - 1
    return vand, cnt

"""
	Python implementation of rule_isset
"""


def rule_isset(tt, ind):
    return tt.bit_test(ind)

"""
	Python implementation of rule_print

	Prints out name, number of samples captured, and percentage
	of samples correctly predicted

	If verbose is set, also prints out the rule's truthtable
"""


def rule_print(r, verbose=False):
    print("Name: {0}, Captured: {1}, Correct: {2}".format(
        r.name, r.captured, r.correct))
    if verbose:
        rule_vector_print(r)
"""
	Python implementation of rule_vector_print
"""


def rule_vector_print(r):
    print("Truthtable: {0}".format(r.tt.digits(2)))

"""
	Python implementation of ruleset_init
"""


def ruleset_init(rules):
    rs = ruleset(len(rules), rules)
    return rs

"""
	Python implementation of ruleset_add
"""


def ruleset_add(rs, rule):
    rs.rules.append(rule)
    rs.tt = rule_vor(rs.tt, rule.tt)
    return rs

"""
	Python implementation of ruleset_delete
"""


def ruleset_delete(rs, rule):
    for i in xrange(rs.nrules):
        if rs.rules[i].name == rule.name:
            rs.rules.pop(i)
    # after deleting a rule from rs, we can no longer use the rs tt
    rs.tt = None
    return rs

"""
	Python implementation of ruleset_print
"""


def ruleset_print(rs):
    for r in rs.rules:
        rule_print(r)

"""
	Count the number of ones in an mpz object
	ensuring we strip off the leading one
"""


def count_ones(tt):
    return gmpy2.popcount(tt) - 1

"""
	Common code for accuracy and check_prefix

	Goes through the rules in a ruleset and counts how many
	samples the ruleset predicts correctly
"""


def count_corr(rs):
    corr = 0
    tot = rs.rules[0].cardinality
    unseen = make_all_ones(tot + 1)
    for r in rs.rules:
        cap = rule_vand(r.tt, unseen)[0]
        corr += rule_vand(cap, labels[r.predict].tt)[1]
        unseen = rule_vandnot(unseen, r.tt)[0]
    return corr, unseen