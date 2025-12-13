# project-work


### Main File Requirements (s347896.py)

1. Define a class responsible for generating the problem and storing the best solution found.
2. Implement a method called solution() that returns the optimal path in the following format: 
```python
[(c1, g1), (c2, g2), …, (cN, gN), (0, 0)]
```
where:
- c1, …, cN represent the sequence of cities visited.
- g1, …, gN represent the corresponding gold collected at each city.

**Scenario:**
- We have N cities with unit square-coordinates
- Variable amount of gold in each city
- Edge weight: geometric distance
- Variable edge density
- There is always one path (graph is always connected even if density is low)

### Goal
- Start in city 0 (base) @ (0.5, 0.5)
- Bring all the gold back to the base.
- You can only leave the gold to the base, no secondary drop cities.
- You don't have to take all the gold from a single city in your run.
- Cost for moving from $i$ to $j$ carrying $g$ gold:
    - $c = d_{ij} + (\alpha \cdot d_{ij} \cdot g)^{\beta}$
    - $a$ & $b$ $\ge 0$

### Notes
- It is not necessary to push the report.pdf or log.pdf in this repo.
- It is mandatory to upload it in "materiale" section of "portale della didattica" at least 168 hours before the exam call.
- For well commented codes, I can't ensure a higher mark but they would be very welcome.
- In case you face any issue or you have any doubt text me at the email giuseppe.esposito@polito.it and professor Squillero giovanni.squillero@polito.it.

	
**Rules (For the Student to worry about):**
- Solutions not performing significantly better than the baseline will get 0 points.
- Solutions will be tested with variable parameters.
- Try to write the core solution algorithm yourself.
