import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probabilities = {}
    for person in people:
        probabilities[person] = 1

    mother_genes, father_genes = 0, 0

    for person in people:
        if people[person]['mother']:
            mother_genes = 1 if people[person]["mother"] in one_gene else 2 if \
                people[person]["mother"] in two_genes else 0
            father_genes = 1 if people[person]["father"] in one_gene else 2 if \
                people[person]["father"] in two_genes else 0

        if  person in one_gene:
            if not people[person]['mother']:
                probabilities[person] *= PROBS["gene"][1]
            else:
                if mother_genes + father_genes == 0 or mother_genes + father_genes == 4:
                    probabilities[person] *= 2 * (PROBS["mutation"] * (1 - PROBS["mutation"]))

                elif mother_genes + father_genes == 1:
                    probabilities[person] *= 1/2

                elif mother_genes + father_genes == 2:
                    if mother_genes == 0 or father_genes == 0:
                        probabilities[person] *= (
                                PROBS["mutation"] * PROBS["mutation"] +
                                (1 - PROBS["mutation"]) * (1- PROBS["mutation"])
                        )
                    else:
                        probabilities[person] *= 1/2

                else:
                    # 3 genes from parents
                    probabilities[person] *= (
                                                (1 - PROBS["mutation"]) * PROBS["mutation"] +
                                                (1/2)*PROBS["mutation"]*PROBS["mutation"] +
                                                (1/2)*(1 - PROBS["mutation"])*(1 - PROBS["mutation"])
                                            )

        elif person in two_genes:
            if not people[person]['mother']:
                probabilities[person] = PROBS["gene"][2]
            else:
                if mother_genes + father_genes == 0:
                    probabilities[person] *= (PROBS["mutation"] * PROBS["mutation"])

                elif mother_genes + father_genes == 1:
                    probabilities[person] *= 1/2*PROBS["mutation"]

                elif mother_genes + father_genes == 2:
                    if mother_genes == 0 or father_genes == 0:
                        probabilities[person] *= (1 - PROBS["mutation"])*PROBS["mutation"]

                    else:
                        probabilities[person] *= 1/4

                elif mother_genes + father_genes == 3:
                    # 3 genes from parents
                    probabilities[person] *= 1/2*(1 - PROBS["mutation"])
                else:
                    probabilities[person] *= (1 - PROBS["mutation"]) *(1 - PROBS["mutation"])

        else:
            if not people[person]['mother']:
                probabilities[person] = PROBS["gene"][0]
            else:
                if mother_genes + father_genes == 0:
                    probabilities[person] *= (1-PROBS["mutation"]) * (1 - PROBS["mutation"])
                elif mother_genes + father_genes == 1:
                    probabilities[person] *= 1 / 2 * (1 - PROBS["mutation"])
                elif mother_genes + father_genes == 2:
                    if mother_genes == 0 or father_genes == 0:
                        probabilities[person] *= (1 - PROBS["mutation"])*PROBS["mutation"]
                    else:
                        probabilities[person] *= 1/4*(1 - PROBS["mutation"])*(1 - PROBS["mutation"]) + \
                        1 / 4 * PROBS["mutation"] * PROBS["mutation"] + \
                        1/4 * PROBS["mutation"]* (1-PROBS["mutation"])

                elif mother_genes + father_genes == 3:
                    # 3 genes from parents
                    probabilities[person] *= PROBS["mutation"]*1/2

                else:
                    probabilities[person] *= PROBS["mutation"]*PROBS["mutation"]

        with_trait =  person in have_trait
        if person in one_gene:
            probabilities[person] *= PROBS["trait"][1][with_trait]
        elif person in two_genes:
            probabilities[person] *= PROBS["trait"][2][with_trait]
        else:
            probabilities[person] *= PROBS["trait"][0][with_trait]


    final_proba = 1
    for _person in people:
        final_proba *= probabilities[_person]
    return  final_proba


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        if probabilities[person]["trait"][False] != 0:
            k = probabilities[person]["trait"][True] / probabilities[person]["trait"][False]
            # 1 = traitTrue + traitFalse = (k+1)traitFalse => traitFalse = 1/(k+1)
            probabilities[person]["trait"][False] = 1 / (k+1)
            probabilities[person]["trait"][True] = k * probabilities[person]["trait"][False]
        else:
            probabilities[person]["trait"][True] = 1

        if probabilities[person]["gene"][2] != 0:
            # 1 = 1G + 2G + 0G = factor1 * 2G + 2G + factor2 * 2G = 2G(factor1+factor2+1)
            # => 2G = 1/(factor1+factor2+1)
            factor1 = probabilities[person]["gene"][1] / probabilities[person]["gene"][2]
            factor2 = probabilities[person]["gene"][0] / probabilities[person]["gene"][2]

            probabilities[person]["gene"][2] = 1/(1+factor2+factor1)
            probabilities[person]["gene"][1] = factor1 * probabilities[person]["gene"][2]
            probabilities[person]["gene"][0] = factor2 * probabilities[person]["gene"][2]
        else:
            if probabilities[person]["gene"][1] != 0:
                k = probabilities[person]["gene"][0]/probabilities[person]["gene"][1]
                probabilities[person]["gene"][1] = 1 / (k+1)
                probabilities[person]["gene"][0] = k * probabilities[person]["gene"][1]

            else:
                probabilities[person]["gene"][0] = 1

if __name__ == "__main__":
    main()
