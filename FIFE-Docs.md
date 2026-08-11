# FIFE Batch Queues

*Workload-level telemetry* - A decision engine compares the Condor queue length with the available pilot slots and then determines when to start new pilot jobs. Pilot requests: 100 pilot jobs, each requesting one whole node for 8 hours on Theta/Cori.

### Dataset summary

| Duration        | Jobs       | Batches    | Users | Sites | Failed    | % Failed |
| --------------- | ---------- | ---------- | ----- | ----- | --------- | -------- |
| 01/2024-06/2024 | 42,720,512 | 18,198,133 | 412   | 46    | 7,638,662 | 17.88    |

*Dataset is not yet publically available, anonymization is underway

### Data Attributes

**Submission path -** `submission_path`

- Production (POMS)
- Development/User

**Provisioning layer -** `provisioning`

- Pilot
- Local
- Glidein payload

**Workflow structure -** `is_dag_manager` , `was_matched` ,`was_held` , `started_executing`

- DAG node/manager
- Standalone

**Lifecycle outcome -** `outcome`

- Completed
- Removed-Started
- Removed Not-Started
- Failed (Application)
- Failed (Submission)
- Failed (Hardware)

## Job Failure Definition

This dataset contains job outcome features, but no failure labels. HTCondor defines job failure as jobs with non-zero exit codes; we additionally count jobs that exited via a user/application-defined code (`ExitCode != 0`) died on a signal (  `ExitSignal` non-null) or were removed from the queue (`JobStatus == 3`). We further classify failure into 3 different types:

1. **Application Failure**: the payload software broke while running. Observable as art/framework processing errors and exceptions (`EventProcessorFailure`, `DataCorruption`, `LogicError`, `ProductNotFound`, uncaught `std::exception`/`bad_alloc`, file open/read errors), crash signals (SIGSEGV, SIGABRT, SIGILL, SIGBUS, SIGFPE, SIGTRAP, SIGPIPE), and DAG nodes removed because a sibling node failed. Resubmitting the identical job reproduces the failure; the fix lives in the code or its input data, not in the submission.
2. **Submission Failure**: the submission or configuration was wrong. Observable as configuration/environment exit codes (bad or missing FHiCL files, broken environment/PATH), timeout kills that trace back to the requested lifetime (exit codes 124/137, fixable via `-Osubmit.timeout` in POMS), manual interrupts and `condor_rm` removals, resource-limit holds, and `SYSTEM_PERIODIC_REMOVE`/`PeriodicRemove` policy removals for exceeding requested limits or sitting held too long. The fix is changing the job request (resources, configuration, inputs) and resubmitting.
3. **Hardware Failure**: the infrastructure failed the job. Observable as graceful-shutdown SIGTERM from node draining / maintenance / glidein time-limit expiration, SIGHUP from a lost session/connection, `JobRouter aborted job` (and orphan) removals, and worker-node environment holds (`FailedToCreateProcess`, `IwdError`, `UnableToInitUserLog`, `SingularityTestFailed`). Nothing about the job needs to change; the same submission may simply succeed on a different node or at a different time.

## Label Construction

Rules apply top-down; the first match wins:

1. `ExitCode == 0`, null `ExitSignal`, and not removed (`JobStatus != 3`) → **Success**. (A removed job can carry `ExitCode == 0`; it is labeled by its `RemoveReason` instead.)
2. Non-null `ExitSignal` → signal table below. Signals take precedence when an `ExitCode` is also present. Signals not in the table → **Application**.
3. Non-zero `ExitCode` → code table below. Codes not in the table → **Application**.
4. Removed with null code & signal → `RemoveReason` rules below; unmatched reasons → **Hardware**.
5. `LastHoldReasonCode` rules are kept for completeness: a job that ends held-then-removed typically also carries a `RemoveReason` or exit code that matches an earlier rule.

### Campaign Type

| **Step** | **Stage Name**                                                                                               | **What it does**                   | **Job Profile**                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | ------------------------------------------- |
| Generation     | `gen, dio, endpoint, corsika, sim, wiremod, ly, offset, spill, g4, beamgun, decay, surface, cryo`                      | Simulates particles hitting the detector | **Heavy** : High CPU, long runtimes   |
| Reconstruction | `stage0, stage1, reco, reco1, reco2, digi, track, decode, recluster, fullproduction, ndlar, compress, convert, fmatch` | Turns raw signals into 3D tracks/hits    | **Medium** : High memory usage        |
| Merging        | `merge, skim, hadd, filter, scrub, watchdog, sleep, test, fclless, concat, mix`                                        | Combines 100 small files into 1 big file | **Light** : Low CPU, fast, mostly I/O |
| Analysis       | `ana, caf, ntuple, larcv`                                                                                              | Runs the actual physics math             | Variable: Depends on user code              |

Matching is case-insensitive substring, against `POMS4_CAMPAIGN_STAGE_NAME`. Jobs with no
`POMS4_*` fields are not part of a campaign and are typed **User**, giving 5 values overall.
`POMS4_CAMPAIGN_TYPE` itself is null for 100% of rows in the raw data and is unusable.

Composite stage names (e.g. `gen_g4_detsim_reco1_reco2_caf`) match several categories at once;
the **rightmost** match wins, on the assumption the terminal step is what the stage delivers
(that example → Analysis). This tie-break decides ~2.98M jobs across 45 composite names.

Note `decay` rather than `run4_decay`: the latter misses `Run5_DecayNoKill` (135,031 jobs).
This table is mirrored by `CAMPAIGN_TYPE_KEYWORDS` in `scripts/attribution-modeling.ipynb`;
keep the two in sync.

#### LastHoldReasonCodes

[HTCondor Docs](https://htcondor.readthedocs.io/en/24.x/codes-other-values/hold-reason-codes.html)

Jobs that are held at any point during execution are given HoldReason/HoldReasonCode, and LastHoldReason/LastHoldReasonCodes are the most recent hold reason given to the job (there could be none or several over the lifetime of the job).

| Code | Reason                            | Meaning                                                                        | Example                                                                                                                                                                                                           | Hold Type  |
| ---- | --------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| 1    | User request                      | The user put the job on hold                                                   | `via condor_hold (by user ...)`                                                                                                                                                                                 | Submission |
| 3    | Job policy                        | The periodic hold expression evaluated to True                                 | `the job attribute PeriodicHold expression ((NumJobStarts > 1) && ((time() - EnteredCurrentStatus) > 1800) evaluated to TRUE)))`                                                                                | Submission |
| 4    | Corrupted credentials             | Credentials for the job are invalid                                            | `Job credentials are not available`                                                                                                                                                                             | Submission |
| 6    | Failed to create process          | The condor starter failed to start the executable                              | `Error running docker job: failed to create shim task: OCI runtime create failed/failed to retrieve spec: unexpected EOF/failed to retrieve labels: unexpected EOF/error reading from server: EOF: unavailable` | Submission |
| 6    | Failed to create process          | The condor starter failed to start the executable                              | `Error running docker job: no space left on device`                                                                                                                                                             | Memory     |
| 6    | Failed to create process          | The condor starter failed to start the executable                              | `Error running docker job: failed to execute '/condor_job_wrapper.sh' with arguments...`                                                                                                                        | Submission |
| 7    | Unable to open output             | The standard output file for the job could not be opened                       | `Failed to open '/': No such file or directory`                                                                                                                                                                 | I/O        |
| 8    | Unable to open input              | The standard input file for the job could not be opened                        |                                                                                                                                                                                                                   | I/O        |
| 12   | Transfer output error             | An error occurred while transferring job output files or self-checkpoint files | `Transfer output files failure at execution point slot_@fnpcXXXX.fnal.gov No such file or directory`                                                                                                            | I/O        |
| 13   | Transfer input error              | An error occurred while transferring job input files                           |                                                                                                                                                                                                                   | I/O        |
| 16   | Spooling input                    | Input files are being spooled                                                  | `Spooling input data files`                                                                                                                                                                                     | I/O        |
| 26   | System policy                     | System periodic hold evaluated to true                                         | `SYSTEM_PERIODIC_HOLD Disk/Memory Limit`                                                                                                                                                                        | Memory     |
| 26   | System policy                     | System periodic hold evaluated to true                                         | `SYSTEM_PERIODIC_HOLD Run Time Limit`                                                                                                                                                                           | Runtime    |
| 32   | Max transfer input size exceeded  | The maximum total input file transfer size was exceeded                        |                                                                                                                                                                                                                   | I/O        |
| 33   | Max transfer output size exceeded | The maximum total output file transfer size was exceeded                       | `Error sending file`                                                                                                                                                                                            | I/O        |
| 34   | Job out of resources              | Job resource usage exceeded the provisioned limit                              | `Docker job has gone over memory limit of XXXX Mb`                                                                                                                                                              | Memory     |
| 35   | Invalid docker image              | Specified docker image was invalid                                             | `Error response from daemon: Requested CPUs are not available`<br /><br />`Error from slot1_37@fnpc22027.fnal.gov:  [blank]`                                                                                  | Submission |

Observed volumes and job failure rates by `LastHoldReasonCode` (01-06/2024; never-held jobs fail at 13.5%):

| Code | Jobs       | Failure rate | Note                                                                                               |
| ---- | ---------- | ------------ | -------------------------------------------------------------------------------------------------- |
| 16   | 20,925,783 | 16.6%        | Routine input spooling — barely above the never-held rate; a lifecycle marker, not a warning sign |
| 26   | 1,154,966  | 99.3%        | Near-deterministic failure once this hold fires                                                    |
| 34   | 807,566    | 46.7%        | about half recover after release                                                                   |
| 35   | 300,517    | 45.1%        | about half recover after release                                                                   |
| 12   | 163,345    | 79.8%        |                                                                                                    |
| 1    | 134,129    | 22.5%        | Most user-held jobs are released and succeed                                                       |
| 3    | 18,144     | 100.0%       | Always fatal                                                                                       |
| 4    | 872        | 99.0%        | Rare, fatal                                                                                        |
| 6    | 37         | 32.4%        | Rare, not fatal                                                                                    |
| 33   | 7          | 85.7%        | Rare, fatal                                                                                        |
| 7    | 3          | 100.0%       | Rare, fatal                                                                                        |

#### ExitCodes

[SLUR](https://slurm.schedmd.com/job_exit_code.html)[M docs](https://slurm.schedmd.com/job_exit_code.html)

| ExitCode | Name                                    | Meaning(s)                                                                                                                                                                           | Failure Type |
| -------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| 0        | Success                                 |                                                                                                                                                                                      | -            |
| 1        | OtherArt                                | General failure, early exit due to validation check, other command-line processing error, related to modules/event processing                                                        | Application  |
| 2        | StdException                            | Incorrect use of shell builtins                                                                                                                                                      | Application  |
| 3        | Unknown                                 | Some error in job                                                                                                                                                                    | Application  |
| 4        | BadAlloc                                |                                                                                                                                                                                      | Application  |
| 5        | BadExceptionType                        |                                                                                                                                                                                      | Application  |
| 6        | ProductNotFound                         |                                                                                                                                                                                      | Application  |
| 7        | DictionaryNotFound                      |                                                                                                                                                                                      | Application  |
| 8        | InsertFailure                           |                                                                                                                                                                                      | Application  |
| 9        | Configuration                           |                                                                                                                                                                                      | Submission   |
| 10       | LogicError                              | Error is due to missing/unusable data products, and reproducible when being ran interactively                                                                                        | Application  |
| 11       | UnimplementedFeature                    |                                                                                                                                                                                      | Application  |
| 12       | InvalidReference                        |                                                                                                                                                                                      | Application  |
| 13       | TypeConversion                          |                                                                                                                                                                                      | Application  |
| 14       | NullPointerError                        |                                                                                                                                                                                      | Application  |
| 15       | EventTimeout                            |                                                                                                                                                                                      | Application  |
| 16       | DataCorruption                          |                                                                                                                                                                                      | Application  |
| 17       | ScheduleExecutionFailure                |                                                                                                                                                                                      | Application  |
| 18       | EventProcessorFailure                   |                                                                                                                                                                                      | Application  |
| 19       | EndJobFailure                           |                                                                                                                                                                                      | Application  |
| 20       | FileOpenError                           | network/xrootd error, error is due to network issue causing input file cannot be found/opened                                                                                        | Application  |
| 21       | FileReadError                           |                                                                                                                                                                                      | Application  |
| 22       | FatalRootError                          |                                                                                                                                                                                      | Application  |
| 23       | MismatchedInputFiles                    |                                                                                                                                                                                      | Application  |
| 24       | CatalogServiceError                     |                                                                                                                                                                                      | Application  |
| 25       | ProductDoesNotSupportViews              |                                                                                                                                                                                      | Application  |
| 26       | ProductDoesNotSupportPtr                |                                                                                                                                                                                      | Application  |
| 27       | SQLExecutionError                       |                                                                                                                                                                                      | Application  |
| 28       | InvalidNumber                           |                                                                                                                                                                                      | Application  |
| 29       | NotFound                                |                                                                                                                                                                                      | Application  |
| 30       | ServiceNotFound                         |                                                                                                                                                                                      | Application  |
| 31       | ProductCannotBeAggregated               |                                                                                                                                                                                      | Application  |
| 32       | ProductRegistrationFailure              |                                                                                                                                                                                      | Application  |
| 33       | EventRangeOverlap                       |                                                                                                                                                                                      | Application  |
| 35       | unidentified                            |                                                                                                                                                                                      | Application  |
| 44       | unidentified                            |                                                                                                                                                                                      | Application  |
| 52       | unidentified                            |                                                                                                                                                                                      | Application  |
| 54       | unidentified                            |                                                                                                                                                                                      | Application  |
| 65       | cet::exception                          | Due to missing input files, e.g. missing flux files when running neutrino interaction                                                                                                | Submission   |
| 66       | std::exception                          |                                                                                                                                                                                      | Application  |
| 67       | unidentified                            |                                                                                                                                                                                      | Application  |
| 68       | std::bad_alloc                          |                                                                                                                                                                                      | Application  |
| 69       | unidentified                            |                                                                                                                                                                                      | Application  |
| 70       | std::exception                          |                                                                                                                                                                                      | Application  |
| 71       | unidentified                            |                                                                                                                                                                                      | Application  |
| 73       | unidentified                            |                                                                                                                                                                                      | Application  |
| 74       | unidentified                            |                                                                                                                                                                                      | Application  |
| 88       | unidentified                            |                                                                                                                                                                                      | Application  |
| 89       | unidentified                            |                                                                                                                                                                                      | Application  |
| 90       | FHiCL configuration parse error         | Due to missing FHiCL file, e.g. typos in the file name or the full path to the directory where the FCL file is located (only for FHiCL files that haven't been pushed to icaruscode) | Submission   |
| 91       | FHiCL ParameterSet generation error     | Due to types/undeclared FHiCL parameters                                                                                                                                             | Submission   |
| 99       | unidentified                            |                                                                                                                                                                                      | Application  |
| 110      | unidentified                            |                                                                                                                                                                                      | Application  |
| 111      | unidentified                            |                                                                                                                                                                                      | Application  |
| 112      | unidentified                            |                                                                                                                                                                                      | Application  |
| 113      | unidentified                            |                                                                                                                                                                                      | Application  |
| 124      |                                         |                                                                                                                                                                                      | Submission   |
| 126      |                                         | Command cannot execute                                                                                                                                                               | Submission   |
| 127      |                                         | Command not found (broken environment/PATH or missing executable in the submission)                                                                                                  | Submission   |
| 129      | 128+1: died on SIGHUP                   | hangup / lost session                                                                                                                                                                | Hardware     |
| 130      | 128+2: died on SIGINT                   | interrupted                                                                                                                                                                          | Submission   |
| 131      | 128+3: SIGQUIT                          | quit from keyboard                                                                                                                                                                   | Submission   |
| 132      | 128+4: died on SIGILL                   | illegal instruction                                                                                                                                                                  | Application  |
| 133      | 128+5: died on SIGTRAP                  | trace/breakpoint trap                                                                                                                                                                | Application  |
| 134      | 128+6: died on SIGABRT                  | the program aborted itself                                                                                                                                                           | Application  |
| 135      | 128+7: died on SIGBUS                   | bus error                                                                                                                                                                            | Application  |
| 136      | 128+8: died on SIGFPE                   | fatal arithmetic error (divide by zero)                                                                                                                                              | Application  |
| 137      | 128+9: SIGKILL signal                  | Program cannot be killed by a SIGTERM signal, and instead is forcefully killed by SIGKILL, can be fixed by changing the -Osubmit.timeout parameter in POMS                           | Submission   |
| 139      | 128+11: died on SIGSEGV                 | segmentation fault                                                                                                                                                                   | Application  |
| 141      | 128+13: died on SIGPIPE                 | writing to a closed/broken pipe                                                                                                                                                     | Application  |
| 143      | 128+15: died on SIGTERM                 | batch system is requesting a graceful shutdown, this commonly happnes during a node draining, a time-limit expiration, or a node maintenance event                                   | Hardware     |
| 201      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 202      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 204      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 222      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 243      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 245      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 247      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 249      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 250      | unidentified wrapper/site-specific code |                                                                                                                                                                                      | Application  |
| 255      | exit(-1) / generic fatal error          |                                                                                                                                                                                      | Application  |

#### Other ExitSignals

*not covered by the codes 128+n above but exist in the Batch Queue logs

| ExitSignal | Meaning(s) | Failure Type |
| ---------- | ---------- | ------------ |
| 74         |            | Application  |
| 76         |            | Application  |
| 94         |            | Application  |
| 117        |            | Application  |
| 121        |            | Application  |
| 122        |            | Application  |
| 124        |            | Application  |
