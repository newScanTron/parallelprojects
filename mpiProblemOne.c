Last login: Thu Apr 21 13:21:14 on ttys002
wl-7-179:~ chrismurphy$ ls
Applications		Music			bisonproto6
AudioKit-iOS		Pictures		clientSide
Desktop			Public			endlessly-working-title
Documents		Qt5.5.1			intel
Downloads		Sites			packs
Library			TextFinder		what
Movies			VST3 SDK		yeah.sql
wl-7-179:~ chrismurphy$ cd documents
wl-7-179:documents chrismurphy$ ls
About Stacks.pdf
AudioKit
AudioKit-iOS
CookingGame
Examples
FlexMonkey
MyPlayground.playground
QuestionAdder
QuizQuestions
SelectorPickerView
Swiftis
SynthStyle
Terminal Saved Output
Uml2CodeTool
ammoliteSite
angular-phonecat
astro_notes.txt
audioExample
build-fileOutPutTest-Desktop_Qt_5_5_0_clang_64bit-Debug
build-uCode-Desktop_Qt_5_5_1_clang_64bit-Debug
build-uCode-Desktop_Qt_5_5_1_clang_64bit-Release
chris_murphy_cuda1
chris_murphy_cuda1.zip
chris_murphy_cuda2
chris_murphy_cuda2.zip
chris_murphy_cuda4
chris_murphy_cuda4.zip
cognitiveChrisMurphy.docx
congnitiveChrisMurphy
congnitiveChrisMurphy.doc
cs344
dudeStuff
fabTemp
fabricon
fileOutPutTest
foo.tex
gameGame
harmonize
helloAudioKit3
home
homeWorkThree
json
lab7
mpitutorial
parallel
reduce_avg.c
refAppProto
side ship.png
site-creator-py
sprites
status_retport_3_24
status_retport_3_24.zip
status_update02_25_chris_murphy
status_update02_25_chris_murphy.zip
tempMainTower.png
test.uct
towerdefense
uml_overview.png
uml_overview_revised.pdf
uml_visitor_strategy.png
useCaseDiagramTowerDefense.png
useCaseDiagramTowerDefense.uxf
wl-7-179:documents chrismurphy$ code parallel
wl-7-179:documents chrismurphy$ ls
About Stacks.pdf
AudioKit
AudioKit-iOS
CookingGame
Examples
FlexMonkey
MyPlayground.playground
QuestionAdder
QuizQuestions
SelectorPickerView
Swiftis
SynthStyle
Terminal Saved Output
Uml2CodeTool
ammoliteSite
angular-phonecat
astro_notes.txt
audioExample
build-fileOutPutTest-Desktop_Qt_5_5_0_clang_64bit-Debug
build-uCode-Desktop_Qt_5_5_1_clang_64bit-Debug
build-uCode-Desktop_Qt_5_5_1_clang_64bit-Release
chris_murphy_cuda1
chris_murphy_cuda1.zip
chris_murphy_cuda2
chris_murphy_cuda2.zip
chris_murphy_cuda4
chris_murphy_cuda4.zip
cognitiveChrisMurphy.docx
congnitiveChrisMurphy
congnitiveChrisMurphy.doc
cs344
dudeStuff
fabTemp
fabricon
fileOutPutTest
foo.tex
gameGame
harmonize
helloAudioKit3
home
homeWorkThree
json
lab7
mpitutorial
parallel
reduce_avg.c
refAppProto
side ship.png
site-creator-py
sprites
status_retport_3_24
status_retport_3_24.zip
status_update02_25_chris_murphy
status_update02_25_chris_murphy.zip
tempMainTower.png
test.uct
towerdefense
uml_overview.png
uml_overview_revised.pdf
uml_visitor_strategy.png
useCaseDiagramTowerDefense.png
useCaseDiagramTowerDefense.uxf
wl-7-179:documents chrismurphy$ code mpitutorials
wl-7-179:documents chrismurphy$ code mpitutorial
wl-7-179:documents chrismurphy$ cmurphy@hpc.mtech.edu
-bash: cmurphy@hpc.mtech.edu: command not found
wl-7-179:documents chrismurphy$ ssh cmurphy@hpc.mtech.edu


Montana Tech of the University of Montana

IMPORTANT NOTICE:

This system is the property of Montana Tech and is subject to the acceptable
use policy located at http://cs.mtech.edu/aup/. Unauthorized use is a violation
of 45-6-311 MCA and MUS policies. By continuing to use this system you indicate
your awareness of and consent to these terms and conditions of use. Log off
immediately if you do not agree to the conditions stated in this warning.

cmurphy@hpc.mtech.edu's password:
Last login: Wed Apr 20 18:05:01 2016 from 69.145.170.10
 _                            _            _               _
| |                          | |          | |             | |
| |__  _ __   ___   _ __ ___ | |_ ___  ___| |__    ___  __| |_   _
| '_ \| '_ \ / __| | '_ ` _ \| __/ _ \/ __| '_ \  / _ \/ _` | | | |
| | | | |_) | (__ _| | | | | | ||  __/ (__| | | ||  __/ (_| | |_| |
|_| |_| .__/ \___(_)_| |_| |_|\__\___|\___|_| |_(_)___|\__,_|\__,_|
      | |
      |_|

See the wiki to learn more about the cluster
http://cs.mtech.edu/wiki/index.php/Category:HPC

See the Ganglia page for detailed cluster status
http://hpc.mtech.edu/ganglia/

Please report any issues or suggestions to bdeng@mtech.edu or jbraun@mtech.edu
[cmurphy@scyld ~]$ ls
avg               hellojob.pbs        hist                 mat
barrier.cu        hellojob.pbs.e4990  histogram.c          matmul2.c
count             hellojob.pbs.o4990  ho                   matmul.c
count_sort.c      hellompi            Job                  mendal
five.2.c          hello_mpi2.c        Job.e4986            omp_trap1.c
h                 hello_mpi3.c        Job.o4986            omp_trap1_orig.c
hello             hello_mpi4.c        linked.c             Problem Set 2
hello.c           hello_mpi.c         linked_omp3_tasks.c  prod_cons.c
helloCuda.cu      helloworld.1.c      linkedTasks.c        reduce_avg.c
helloHost         helloworld.c        mandel               trap
hello_hostname.c  hi.out              mandel.c
[cmurphy@scyld ~]$ vi text.txt
[cmurphy@scyld ~]$ vi text.txt
[cmurphy@scyld ~]$ mpirun -np 10 ./avg 60 text.txt
Retrieved line of length 1123 :
34.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.343434.5, 444.5, 0.55, 2.5354, 2.3434
Local sum for process 0 - 30.263384, avg = 0.504390
Local sum for process 1 - 29.267923, avg = 0.487799
Local sum for process 3 - 28.113129, avg = 0.468552
Local sum for process 4 - 32.184361, avg = 0.536406
Local sum for process 2 - 29.940310, avg = 0.499005
Local sum for process 9 - 27.808136, avg = 0.463469
Local sum for process 8 - 29.772738, avg = 0.496212
Local sum for process 7 - 29.542231, avg = 0.492371
Local sum for process 6 - 28.458265, avg = 0.474304
Local sum for process 5 - 31.720816, avg = 0.528680
Total sum = 297.071289, avg = 0.495119
[cmurphy@scyld ~]$ vi text.txt
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ exit
logout
Connection to hpc.mtech.edu closed.
wl-7-179:documents chrismurphy$ cd ..
wl-7-179:~ chrismurphy$ ls
Applications		Music			bisonproto6
AudioKit-iOS		Pictures		clientSide
Desktop			Public			endlessly-working-title
Documents		Qt5.5.1			intel
Downloads		Sites			packs
Library			TextFinder		what
Movies			VST3 SDK		yeah.sql
wl-7-179:~ chrismurphy$ cd documents
wl-7-179:documents chrismurphy$ ls
About Stacks.pdf
AudioKit
AudioKit-iOS
CookingGame
Examples
FlexMonkey
MyPlayground.playground
QuestionAdder
QuizQuestions
SelectorPickerView
Swiftis
SynthStyle
Terminal Saved Output
Uml2CodeTool
ammoliteSite
angular-phonecat
astro_notes.txt
audioExample
build-fileOutPutTest-Desktop_Qt_5_5_0_clang_64bit-Debug
build-uCode-Desktop_Qt_5_5_1_clang_64bit-Debug
build-uCode-Desktop_Qt_5_5_1_clang_64bit-Release
chris_murphy_cuda1
chris_murphy_cuda1.zip
chris_murphy_cuda2
chris_murphy_cuda2.zip
chris_murphy_cuda4
chris_murphy_cuda4.zip
cognitiveChrisMurphy.docx
congnitiveChrisMurphy
congnitiveChrisMurphy.doc
cs344
dudeStuff
fabTemp
fabricon
fileOutPutTest
foo.tex
gameGame
harmonize
helloAudioKit3
home
homeWorkThree
json
lab7
mpitutorial
parallel
reduce_avg.c
refAppProto
side ship.png
site-creator-py
sprites
status_retport_3_24
status_retport_3_24.zip
status_update02_25_chris_murphy
status_update02_25_chris_murphy.zip
tempMainTower.png
test.uct
towerdefense
uml_overview.png
uml_overview_revised.pdf
uml_visitor_strategy.png
useCaseDiagramTowerDefense.png
useCaseDiagramTowerDefense.uxf
wl-7-179:documents chrismurphy$ cd uml2codetool
wl-7-179:uml2codetool chrismurphy$ ls
LICENSE.md	docs		test		uCode.pro.user
README.md	src		uCode.pro	ui
wl-7-179:uml2codetool chrismurphy$ git status
On branch save_arrows
nothing to commit, working directory clean
wl-7-179:uml2codetool chrismurphy$ git checkout master
Switched to branch 'master'
Your branch is behind 'origin/master' by 40 commits, and can be fast-forwarded.
  (use "git pull" to update your local branch)
wl-7-179:uml2codetool chrismurphy$ git pull
remote: Counting objects: 33, done.
remote: Compressing objects: 100% (28/28), done.
remote: Total 33 (delta 9), reused 4 (delta 4), pack-reused 1
Unpacking objects: 100% (33/33), done.
From https://github.com/UM-ADV16/Uml2CodeTool
   14890eb..55aefaa  master     -> origin/master
 * [new branch]      ayechanBranch -> origin/ayechanBranch
   4ca8021..e31c748  save_load_functions -> origin/save_load_functions
Updating cc16bdd..55aefaa
Fast-forward
 docs/resources/uml_overview.uxf                            | 704 +++++++++++++------------
 docs/resources/uml_visitor_strategy.uxf                    | 416 ++++++++-------
 .../qtquick2controlsapplicationviewer.cpp                  | 102 ++++
 .../qtquick2controlsapplicationviewer.h                    |  28 +
 .../qtquick2controlsapplicationviewer.pri                  | 202 +++++++
 src/UiEventDispatcher.cpp                                  | 277 +++++++++-
 src/UiEventDispatcher.h                                    |  29 +-
 src/main.cpp                                               |   4 +
 src/uBaseClass.cpp                                         |  13 +
 src/uBaseClass.h                                           |   4 +-
 src/uButton.h                                              |   1 +
 src/uChildButton.cpp                                       |   6 +
 src/uChildButton.h                                         |   1 +
 src/uChildClass.cpp                                        |  14 +
 src/uChildClass.h                                          |   3 +
 src/uClassButton.cpp                                       |   9 +
 src/uClassButton.h                                         |   2 +-
 src/uClassDiagram.cpp                                      |  29 +-
 src/uClassDiagram.h                                        |   4 +
 src/uClassFactory.cpp                                      |  37 +-
 src/uClassFactory.h                                        |   1 +
 src/uCodeGenerationVisitor.cpp                             | 188 ++++++-
 src/uCodeGenerationVisitor.h                               |  13 +-
 src/uGridArrow.cpp                                         |   2 +
 src/uGridLayout.cpp                                        |  33 +-
 src/uGridLayout.h                                          |  11 +-
 src/uGridObject.h                                          |   2 +-
 src/uGridSegment.h                                         |   1 +
 src/uInheritable.cpp                                       |   6 +
 src/uInheritable.h                                         |   5 +-
 src/uInterface.cpp                                         |  14 +
 src/uInterface.h                                           |   3 +
 src/uInterfaceButton.cpp                                   |   6 +-
 src/uInterfaceButton.h                                     |   1 +
 src/uStringConverter.cpp                                   |  10 +-
 src/uVisitor.h                                             |   3 +
 test/uAggregationTest.cpp                                  |   1 -
 ui/UClassPanel.qml                                         |   2 +
 ui/UCodeGenerationDialogue.qml                             |  32 +-
 ui/UDrawingCanvas.qml                                      |   1 +
 ui/UFileDialog.qml                                         |   4 +-
 ui/UMenuBar.qml                                            |  35 +-
 ui/USaveFileDialog.qml                                     |  99 ++++
 ui/filesavedialog.cpp                                      | 199 +++++++
 ui/filesavedialog.h                                        |  79 +++
 ui/qml.qrc                                                 |   1 +
 46 files changed, 2042 insertions(+), 595 deletions(-)
 create mode 100644 qtquick2controlsapplicationviewer/qtquick2controlsapplicationviewer.cpp
 create mode 100644 qtquick2controlsapplicationviewer/qtquick2controlsapplicationviewer.h
 create mode 100644 qtquick2controlsapplicationviewer/qtquick2controlsapplicationviewer.pri
 create mode 100644 ui/USaveFileDialog.qml
 create mode 100644 ui/filesavedialog.cpp
 create mode 100644 ui/filesavedialog.h
wl-7-179:uml2codetool chrismurphy$ git pull
Already up-to-date.
wl-7-179:uml2codetool chrismurphy$
  [Restored Apr 21, 2016, 5:35:57 PM]
Last login: Thu Apr 21 17:35:47 on console
Restored session: Thu Apr 21 17:32:04 MDT 2016
ChrisMurphysMBP:uml2codetool chrismurphy$ git checkout save_load_functions
Switched to branch 'save_load_functions'
Your branch is behind 'origin/save_load_functions' by 2 commits, and can be fast-forwarded.
  (use "git pull" to update your local branch)
ChrisMurphysMBP:uml2codetool chrismurphy$ git pull
remote: Counting objects: 1, done.
remote: Total 1 (delta 0), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (1/1), done.
From https://github.com/UM-ADV16/Uml2CodeTool
   55aefaa..30d71ab  master     -> origin/master
Updating 4ca8021..e31c748
Fast-forward
 .../qtquick2controlsapplicationviewer.cpp                  | 102 +++++++++++++
 .../qtquick2controlsapplicationviewer.h                    |  28 ++++
 .../qtquick2controlsapplicationviewer.pri                  | 202 +++++++++++++++++++++++++
 src/UiEventDispatcher.cpp                                  |   4 +-
 src/main.cpp                                               |   4 +
 src/uClassDiagram.cpp                                      |   3 +-
 src/uCodeGenerationVisitor.cpp                             |  24 ++-
 src/uCodeGenerationVisitor.h                               |   1 +
 src/uGridLayout.cpp                                        |   8 +-
 uCode.pro                                                  |  12 +-
 ui/UClassPanel.qml                                         |  63 --------
 ui/UCodeGenerationDialogue.qml                             |   4 +-
 ui/ULoadFileDialog.qml                                     |  73 ---------
 ui/UMenuBar.qml                                            |  18 ++-
 ui/USaveFileDialog.qml                                     |  99 ++++++++++++
 ui/filesavedialog.cpp                                      | 199 ++++++++++++++++++++++++
 ui/filesavedialog.h                                        |  79 ++++++++++
 ui/qml.qrc                                                 |   2 +-
 18 files changed, 770 insertions(+), 155 deletions(-)
 create mode 100644 qtquick2controlsapplicationviewer/qtquick2controlsapplicationviewer.cpp
 create mode 100644 qtquick2controlsapplicationviewer/qtquick2controlsapplicationviewer.h
 create mode 100644 qtquick2controlsapplicationviewer/qtquick2controlsapplicationviewer.pri
 delete mode 100644 ui/ULoadFileDialog.qml
 create mode 100644 ui/USaveFileDialog.qml
 create mode 100644 ui/filesavedialog.cpp
 create mode 100644 ui/filesavedialog.h
ChrisMurphysMBP:uml2codetool chrismurphy$ git status
On branch save_load_functions
Your branch is up-to-date with 'origin/save_load_functions'.
nothing to commit, working directory clean
ChrisMurphysMBP:uml2codetool chrismurphy$ git status
On branch save_load_functions
Your branch is up-to-date with 'origin/save_load_functions'.
nothing to commit, working directory clean
ChrisMurphysMBP:uml2codetool chrismurphy$ git pull
Already up-to-date.
ChrisMurphysMBP:uml2codetool chrismurphy$ ssh cmurphy@hpc.mtech.edu


Montana Tech of the University of Montana

IMPORTANT NOTICE:

This system is the property of Montana Tech and is subject to the acceptable
use policy located at http://cs.mtech.edu/aup/. Unauthorized use is a violation
of 45-6-311 MCA and MUS policies. By continuing to use this system you indicate
your awareness of and consent to these terms and conditions of use. Log off
immediately if you do not agree to the conditions stated in this warning.

cmurphy@hpc.mtech.edu's password:
Last login: Thu Apr 21 16:13:12 2016 from 10.17.7.179
 _                            _            _               _
| |                          | |          | |             | |
| |__  _ __   ___   _ __ ___ | |_ ___  ___| |__    ___  __| |_   _
| '_ \| '_ \ / __| | '_ ` _ \| __/ _ \/ __| '_ \  / _ \/ _` | | | |
| | | | |_) | (__ _| | | | | | ||  __/ (__| | | ||  __/ (_| | |_| |
|_| |_| .__/ \___(_)_| |_| |_|\__\___|\___|_| |_(_)___|\__,_|\__,_|
      | |
      |_|

See the wiki to learn more about the cluster
http://cs.mtech.edu/wiki/index.php/Category:HPC

See the Ganglia page for detailed cluster status
http://hpc.mtech.edu/ganglia/

Please report any issues or suggestions to bdeng@mtech.edu or jbraun@mtech.edu
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c
[cmurphy@scyld ~]$ vi reduce_avg.c

//fprintf(stderr, "this should print\n");
  int num_elements_per_proc = atoi(argv[1]);
//fprintf(stderr, "this should print also\n");

  MPI_Init(&argc, &argv);
//fprintf(stderr, "this should print\n");

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
//fprintf(stderr, "this should print %d\n",world_rank);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//fprintf(stderr, "this should print\n");

  // Create a random array of elements on all processes.
  srand(time(NULL)*world_rank);   // Seed the random number generator to get different results each time for each processor
  float *rand_nums = NULL;
 // rand_nums = create_rand_nums(num_elements_per_proc);
        rand_nums = nums;
 //fprintf(stderr, "we got right before the first for lopp %d\n", world_size);
  // Sum the numbers locally
  float local_sum = 0;
  int i;
  for (i = 0; i < num_elements_per_proc; i++) {
    local_sum += rand_nums[i];
  }

  // Print the random numbers on each process
  printf("Local sum for process %d - %f, avg = %f\n",
         world_rank, local_sum, local_sum / num_elements_per_proc);

  // Reduce all of the local sums into the global sum
  float global_sum;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  // Print the result
  if (world_rank == 0) {
    printf("Total sum = %f, avg = %f\n", global_sum,
           global_sum / (world_size * num_elements_per_proc));
  }

  // Clean up
  free(rand_nums);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
                                                                          181,1         Bot
