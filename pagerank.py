import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    tramod = {}
    linked_pages = [x for x in corpus[page]]
    
    for next_page in corpus.keys():
        if next_page in linked_pages:
            if next_page in tramod.keys():
                tramod[next_page] = tramod[next_page] + damping_factor * (1 / len(linked_pages)) + \
                                      (1 - damping_factor) * (1 / len(corpus.keys()))
            else:
                tramod[next_page] = damping_factor * (1 / len(corpus[page])) + \
                                      (1 - damping_factor) * (1 / len(corpus.keys()))
        else:
            if next_page in tramod.keys():
                tramod[next_page] = (1 - damping_factor) * (1 / len(corpus.keys()))
            else:
                tramod[next_page] = (1 - damping_factor) * (1 / len(corpus.keys()))

    return tramod


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    tally = {}

    # Get starting point
    start_page = random.sample(corpus.keys(), 1)[0]
    current_page = start_page

    for i in range(n):
        # Record page
        if current_page in tally:
            tally[current_page] += 1
        else:
            tally[current_page] = 1

        # Find next page
        transmod = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(list(transmod.keys()), list(transmod.values()))[0]
        current_page = next_page

    # Normalise
    tally = {key: value / n for key, value in tally.items()}

    return(tally)



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Set initial probabilities
    iterprob = {page : 1 / len(corpus) for page in corpus}
    iterchanges = {page : 999 for page in corpus}
    max_iterchange = 999

    # Iterate through to adjust probabilities
    while max_iterchange >= 0.001:
        # Save old dictionary
        old_iterprob = copy.deepcopy(iterprob)

        # Iterate through pages applying formula
        for page in iterprob.keys():
            linked_pages = [ipage for ipage in corpus if page in corpus[ipage]]
            distribution = {k : v for k, v in iterprob.items() if k in linked_pages}
            iterprob[page] = (1 - damping_factor) / len(corpus) + damping_factor * iterative_sum(distribution, corpus)

        # Normalise
        iterprob = {page: (rank / sum(iterprob.values())) for page, rank in iterprob.items()}

        # Set iteration change
        iterchanges = [abs(iterprob[page] - old_iterprob[page]) for page in iterprob.keys()]
        max_iterchange = max(iterchanges)

    return iterprob


def iterative_sum(distribution, corpus):
    s = 0

    for page in distribution:
        s += distribution[page] / len(corpus[page])

    return s


if __name__ == "__main__":
    main()
