import random
import sys
import time

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.crossword.variables:
            self.domains[v] = set([value for value in self.domains[v] if len(value) == v.length])
            # print(f'{v} correspondance {self.domains[v]}')

    def revise(self, x: Variable, y: Variable):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        value_to_remove = []
        common_case = self.crossword.overlaps[x, y]
        # print(f'Common case of {x} and {y} is {common_case}')
        if common_case:
            for value in self.domains[x]:
                anomaly = all(value[common_case[0]] != _value[common_case[1]] for _value in self.domains[y])
                if anomaly:
                    value_to_remove.append(value)
            if value_to_remove:
                self.domains[x] = set(one_value for one_value in self.domains[x] if one_value not in value_to_remove)
                return True
        return False

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        general_queue = []
        if arcs:
            general_queue = arcs.copy()
        else:
            for variable in self.crossword.variables:
                neighbors = self.crossword.neighbors(variable)
                for neighbor in neighbors:
                    general_queue.append((variable, neighbor))

        # print(general_queue)
        while general_queue:
            x, y = general_queue.pop(0)
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                neighbors = self.crossword.neighbors(x)
                for neighbor in neighbors:
                    if neighbor != y:
                        general_queue.append((neighbor, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.crossword.variables:
            if variable not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words_used = []

        for variable in assignment:
            if not len(assignment[variable]) == variable.length:
                return False
            if assignment[variable] in words_used:
                return False
            words_used.append(assignment[variable])

            for y in self.crossword.neighbors(variable):
                if y in assignment:
                    common_case = self.crossword.overlaps[variable, y]
                    if assignment[variable][common_case[0]] != assignment[y][common_case[1]]:
                        return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        map_dictionary = {}
        for value in self.domains[var]:
            map_dictionary[value] = 0

        unassigned_neighbors = set(n for n in self.crossword.neighbors(var) if n not in assignment)
        for value in self.domains[var]:

            for a_neigbor in unassigned_neighbors:
                common_case = self.crossword.overlaps[var, a_neigbor]
                for a_neigbor_value in self.domains[a_neigbor]:
                    if value[common_case[0]] != a_neigbor_value[common_case[1]]:
                        map_dictionary[value] += 1

        # print(map_dictionary)
        return [val for val, count in sorted(map_dictionary.items(), key=lambda item: item[1])]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        dictionnary = {}
        for variable in self.crossword.variables:
            if variable not in assignment:
                dictionnary[variable] = len(self.domains[variable])

        with_min_rem_heur = [_var for _var in dictionnary if dictionnary[_var] == min(dictionnary.values())]

        dictionnary.clear()
        if len(with_min_rem_heur) > 1:
            for remaining in with_min_rem_heur:
                dictionnary[remaining] = len(self.crossword.neighbors(remaining))

            with_largest_degree = [_var for _var in dictionnary if dictionnary[_var] == min(dictionnary.values())]
            if len(with_largest_degree) > 1:
                return random.choice(with_largest_degree)
            else:
                return with_largest_degree[0]
        else:
            return with_min_rem_heur[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment

        new_var = self.select_unassigned_variable(assignment=assignment)
        self.ac3([(new_var, neighboor) for neighboor in self.crossword.neighbors(new_var)])
        ordered_domain = self.order_domain_values(new_var, assignment)

        for value in ordered_domain:
            assignment[new_var] = value
            if self.consistent(assignment):
                status = self.backtrack(assignment)
                if status:
                    return status
            else:
                del assignment[new_var]

        return None


def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()

