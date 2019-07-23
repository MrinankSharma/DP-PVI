# DP-PVI

The Sacred logging functionality relies on the existence of a Mongo database, by default assumed to be in localhost:9001.
If this is on another machine, ssh port forwarding is an easy way to get this working without changing the code.
(E.g. for mlg people, if the database is on Hinton: `ssh hinton -L 9001:localhost:9001` will connect port 9001 on Hinton
to port 9001 on your machine. In order to bring up and bring down the database, suitable scripts are in /scratch/mjh252/mongo)

Omniboard is an excellent way to view results in a Sacred database.

Note: we use `torch=1.1.0` for compatability with Pipenv.
  OSX users may need to install `libomp` e.g. `brew install libomp`
  
Note: for slack notifications, you will need to add a slack webhook into a file name `slack_webhook` in the base project directory. 