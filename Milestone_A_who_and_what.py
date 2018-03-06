#!/usr/bin/python3
'''Milestone_A_who_and_what.py
This runnable file will provide a representation of
answers to key questions about your project in CSE 415.

'''

# DO NOT EDIT THE BOILERPLATE PART OF THIS FILE HERE:

CATEGORIES=['Baroque Chess Agent','Feature-Based Reinforcement Learning for the Rubik Cube Puzzle',\
  'Supervised Learning: Comparing Trainable Classifiers']

class Partner():
  def __init__(self, lastname, firstname, uwnetid):
    self.uwnetid=uwnetid
    self.lastname=lastname
    self.firstname=firstname

  def __lt__(self, other):
    return (self.lastname+","+self.firstname).__lt__(other.lastname+","+other.firstname)

  def __str__(self):
    return self.lastname+", "+self.firstname+" ("+self.uwnetid+")"

class Who_and_what():
  def __init__(self, team, option, title, approach, workload_distribution, references):
    self.team=team
    self.option=option
    self.title=title
    self.approach = approach
    self.workload_distribution = workload_distribution
    self.references = references

  def report(self):
    rpt = 80*"#"+"\n"
    rpt += '''The Who and What for This Submission

Final Project in CSE 415, University of Washington, Winter, 2018
Milestone A

Team: 
'''
    team_sorted = sorted(self.team)
    # Note that the partner whose name comes first alphabetically
    # must do the turn-in.
    # The other partner(s) should NOT turn anything in.
    rpt += "    "+ str(team_sorted[0])+" (the partner who must turn in all files in Catalyst)\n"
    for p in team_sorted[1:]:
      rpt += "    "+str(p) + " (partner who should NOT turn anything in)\n\n"

    rpt += "Option: "+str(self.option)+"\n\n"
    rpt += "Title: "+self.title + "\n\n"
    rpt += "Approach: "+self.approach + "\n\n"
    rpt += "Workload Distribution: "+self.workload_distribution+"\n\n"
    rpt += "References: \n"
    for i in range(len(self.references)):
      rpt += "  Ref. "+str(i+1)+": "+self.references[i] + "\n"

    rpt += "\n\nThe information here indicates that the following file will need\n"+\
     "to be submitted (in addition to code and possible data files):\n"
    rpt += "    "+\
     {'1':"Baroque_Chess_Agent_Report",'2':"Rubik_Cube_Solver_Report",\
      '3':"Trainable_Classifiers_Report"}\
        [self.option]+".pdf\n"

    rpt += "\n"+80*"#"+"\n"
    return rpt

# END OF BOILERPLATE.

# Change the following to represent your own information:

bryan = Partner("Quah", "Bryan", "bryantpq")
matthew = Partner("Xu", "Matthew", "mattxu")
team = [bryan, matthew]

OPTION = '3'
# Legal options are 1, 2, and 3.

title = ""
 # In this case, the Python file for the formulation would be named End_Poverty.py.

approach = '''We will begin by compiling our data sets that we will be using - CIFAR-10 & 
CS:GO Competitive Matchmaking Data. Next, we will study how to implement our classifiers,
Random Forests & Neural Networks, using our listed references and also learn how to use
common libraries for future use. We will implement a command-line interface for users
to select the classifier they want to use on a selected data set for easy program usage.
'''

workload_distribution = '''Both of us will work together to implement both classifiers
as we feel it is important to deeply understand how they work. We will also both work on
implementing either bagging or boosting after we have finished our classifiers.'''

reference1 = '''Scikit-Learn writeup on Neural Networks and Random Forests;
    URL: http://scikit-learn.org/stable/modules/neural_networks_supervised.html
    URL: http://scikit-learn.org/stable/modules/ensemble.html'''

reference2 = '''Notes from Stanford University's CS class - CS231n: Convolutional Neural
        Networks for Visual Recognition.
    URL: http://cs231n.github.io/'''

our_submission = Who_and_what([bryan, matthew], OPTION, title, approach, workload_distribution, [reference1, reference2])

# You can run this file from the command line by typing:
# python3 who_and_what.py

# Running this file by itself should produce a report that seems correct to you.
if __name__ == '__main__':
  print(our_submission.report())
