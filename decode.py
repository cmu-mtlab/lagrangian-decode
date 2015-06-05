#!/usr/bin/env python
import argparse
import sys
import models
import Queue
from collections import namedtuple, defaultdict

class Edge:
	def __init__(self, tail, head, score, span, english):
		self.tail = tail
		self.head = head
		self.score = score
		self.span = span
		self.english = english

	def __str__(self):
		return '[%d, %d) --> %s' % (self.span[0], self.span[1], ' '.join(self.english))

	def __repr__(self):
		return str(self)

	def __hash__(self):
		return hash((self.tail, self.head, self.span, self.english))

def edges_from(state, lenf, phrases, lm, penalties, constraints):
	if state.lm_state == None:
		return

	if state.words_translated == lenf:
		cost = -lm.end(state.lm_state)
		to_state = State(None, lenf, None, state.coverage)
		edge = Edge(state, to_state, cost, None, tuple())
		yield edge
		return

	for i, j, english, phrase_score in phrases:
		if not (j <= state.last_contiguous_span[0] or i >= state.last_contiguous_span[1]):
			continue
		coverage = 0
		for index, k in enumerate(constraints):
			if k >= i and k < j:
				coverage |= 2 ** index

		if (coverage & state.coverage) != 0:
			continue

		words_translated = state.words_translated + j - i
		if words_translated > lenf:
			continue
		cost = -phrase_score
		lm_state = state.lm_state

		for word in english:
			lm_state, word_logprob = lm.score(lm_state, word)
			cost += -word_logprob

		for k in range(i, j):
			cost += penalties[k]

		if i == state.last_contiguous_span[1]:
			last_contiguous_span = (state.last_contiguous_span[0], j)
		elif j == state.last_contiguous_span[0]:
			last_contiguous_span = (i, state.last_contiguous_span[1])
		else:
			last_contiguous_span = (i, j)

		to_state = State(lm_state, words_translated, last_contiguous_span, coverage | state.coverage)
		edge = Edge(state, to_state, cost, (i, j), tuple(english))
		yield edge

def dijkstra(start, goal, edge_generator):
	dist = defaultdict(lambda: float('inf'))
	prev = defaultdict(lambda: None)

	Q = Queue.PriorityQueue()
	Q.put((0, start))

	while not Q.empty():
		k, u = Q.get()
		if k <= dist[u]:
			dist[u] = k

			if u == goal:
				break

			if dist[u] == float('inf'):
				# Goal node is not reachable
				break

			for edge in edge_generator(u):
				v = edge.head
				alt = dist[u] + edge.score
				if alt < dist[v]:
					Q.put((alt, v))
					dist[v] = alt
					prev[v] = edge

	S = []
	u = goal
	while prev[u] != None:
		S.append(prev[u])
		u = prev[u].tail
	S.reverse()
	return dist[goal], S

def get_relevant_phrases(f, tm):
	relevant_phrases = set()
	for i in range(len(f)):
		for j in range(i + 1, len(f) + 1):
			french = tuple(f[i:j])
			if french not in tm:
				continue
			for phrase in tm[french]:
				relevant_phrases.add((i, j, tuple(phrase.english.split()), phrase.logprob))
	return relevant_phrases

def decode(f, tm, lm, max_iterations=sys.maxint):
	start_state = State(lm.begin(), 0, (0, 0), 0)	

	relevant_phrases = get_relevant_phrases(f, tm)
	penalties = [0.0 for _ in f]
	constraints = []

	prev_score = float('inf')
	lambdaa = 0
	iteration = 0
	while iteration < max_iterations:
		iteration += 1
		print >>sys.stderr, 'Decoding (iteration %d)...' % iteration
		goal_state = State(None, len(f), None, (2 ** len(constraints) - 1))
		edge_generator=lambda state: edges_from(state, len(f), relevant_phrases, lm, penalties, constraints)
		score, path = dijkstra(start_state, goal_state, edge_generator)
		if -score > prev_score:
			lambdaa += 1
			print >>sys.stderr, 'Score went up -- decreasing alpha'
		prev_score = -score

		english = []
		usage = [0 for _ in f]
		for edge in path:
			if edge.span == None:
				continue
			for i in range(*edge.span):
				usage[i] += 1
			english += list(edge.english)

		print '%f\t%s' % (-score, ' '.join(english)), [x for x in list(enumerate(usage)) if x[1] != 0]

		if False not in [usage[i] == 1 for i in range(len(f))]:
			print 'Optimal solution found!'
			break
		else:
			for i in range(len(f)):
				penalties[i] += (usage[i] - 1) * alpha(lambdaa)

		if iteration % 10 == 0:
			ranked_usage = sorted(enumerate(usage), key=lambda (i, c): c, reverse=True)
			possible_constraints = [i for i, c in ranked_usage if i not in constraints and (i + 1) not in constraints and (i - 1) not in constraints and usage[i] > 1]
			if len(possible_constraints) > 0:
				most_overused = possible_constraints[0]
			else:
				possible_constraints = [i for i, c in ranked_usage if i not in constraints and usage[i] > 1]
				assert len(possible_constraints) > 0
				most_overused = possible_constraints[0]
			print >>sys.stderr, 'Still not converged... adding constraint on word %d' % most_overused
			constraints.append(most_overused)

# coverage is a bit vector reprsenting which words are covered
# only words with hard constraints are reprsented in coverage.
State = namedtuple('State', 'lm_state, words_translated, last_contiguous_span, coverage')

parser = argparse.ArgumentParser(description='Simple phrase based decoder.')
parser.add_argument('-i', '--input', dest='input', default='data/input', help='File containing sentences to translate (default=data/input)')
parser.add_argument('-t', '--translation-model', dest='tm', default='data/tm', help='File containing translation model (default=data/tm)')
parser.add_argument('-n', '--num_sentences', dest='num_sents', default=sys.maxint, type=int, help='Number of sentences to decode (default=no limit)')
parser.add_argument('-l', '--language-model', dest='lm', default='data/lm', help='File containing ARPA-format language model (default=data/lm)')
parser.add_argument('-e', '--exact', action='store_true', help='Enable exact decoding (very slow!)')
args = parser.parse_args()

tm = models.TM(args.tm, sys.maxint)
lm = models.LM(args.lm)
sys.stderr.write('Decoding %s...\n' % (args.input,))
input_sents = [tuple(line.strip().split()) for line in open(args.input).readlines()[:args.num_sents]]
alpha = lambda lambdaa: 1.0 / (1.0 + lambdaa)

for f in input_sents:
	print >>sys.stderr, 'Got input sentence "%s"' % ' '.join(f)
	decode(f, tm, lm)
