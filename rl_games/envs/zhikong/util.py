
obs_feature_list = ['position/h-sl-ft', 'attitude/pitch-rad', 'attitude/roll-rad',
                    'attitude/psi-deg', 'aero/beta-deg', 'position/lat-geod-deg', 
                    'position/long-gc-deg', 
                    'velocities/u-fps', 'velocities/v-fps', 'velocities/w-fps', 
                    'velocities/v-north-fps', 'velocities/v-east-fps', 
                    'velocities/v-down-fps', 'velocities/p-rad_sec', 
                    'velocities/q-rad_sec', 'velocities/r-rad_sec', 
                    'fcs/left-aileron-pos-norm', 'fcs/right-aileron-pos-norm', 
                    'fcs/elevator-pos-norm', 'fcs/rudder-pos-norm', 
                    'fcs/throttle-pos-norm',
                    'velocities/ve-fps', 'velocities/h-dot-fps', 'velocities/mach', 
                    'forces/load-factor', 'LifeCurrent', 'BulletCurrentNum',
                    'TargetIntoView', 'AllyIntoView', 'TargetEnterAttackRange', 
                    'AimMode', 'SRAAMCurrentNum', 'SRAAM1_CanReload', 'SRAAM2_CanReload', 
                    'SRAAMTargetLocked', 'AMRAAMCurrentNum', 'AMRAAMCanReload', 
                    'AMRAAMlockedTarget', 'MissileAlert', 'IsOutOfValidBattleArea', 
                    'OutOfValidBattleAreaCurrentDuration', 'DeathEvent']
print(f"len obs: {len(obs_feature_list)}")

act_feature_list = ["fcs/aileron-cmd-norm", "fcs/elevator-cmd-norm",
                    "fcs/throttle-cmd-norm", 
                    "fcs/weapon-launch", "switch-missile",]

hrl_act_feature_list = ["mode", "target_longdeg", "target_latdeg", "target_altitude_ft", "target_velocity",
                        "target_track_deg", "simulation/do_simple_trim", "change-target"]

command_sent = {'red':{'red_0':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                    #    'red_1':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                    #    'red_2':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                    #    'red_3':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                    #    'red_4':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                    #    'red_5':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0}
                       },
                'blue':{'blue_0':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        # 'blue_1':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        # 'blue_2':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        # 'blue_3':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        # 'blue_4':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0},
                        # 'blue_5':{"fcs/aileron-cmd-norm": 0, "fcs/elevator-cmd-norm": 0, "fcs/rudder-cmd-norm": 0, "fcs/throttle-cmd-norm": 0.1,"fcs/weapon-launch":0,"change-target":99,"switch-missile":0}
                       }
                }


def init_info(agent_nums, reset=True):

    assert agent_nums in [1, 2, 3, 4, 5, 6], "agent_nums must between 1~6"
    render = 0
    if agent_nums == 1:
        dict_init = {'flag':{'init':{'render': render}},
                    'red':{
                        'red_0':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.12, "ic/lat-geod-deg": 0.1,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                }
                        },
                    'blue':{
                        'blue_0':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.12, "ic/lat-geod-deg": 0.1,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            }
                            }
                    }

    elif agent_nums == 2:
        dict_init = {'flag':{'init':{'render': render}},
                    'red':{
                        'red_0':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.12, "ic/lat-geod-deg": 0.01,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_1':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.12, "ic/lat-geod-deg": 0.05,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        },
                    'blue':{
                        'blue_0':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.12, "ic/lat-geod-deg": 0.01,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_1':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.12, "ic/lat-geod-deg": 0.05,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                            }
                    }

    elif agent_nums == 3:
        dict_init = {'flag':{'init':{'render': render}},
                    'red':{
                        'red_0':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.12, "ic/lat-geod-deg": 0.01,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_1':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.12, "ic/lat-geod-deg": 0.05,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_2':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.12, "ic/lat-geod-deg": 0.1,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        },
                    'blue':{
                        'blue_0':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.12, "ic/lat-geod-deg": 0.01,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_1':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.12, "ic/lat-geod-deg": 0.05,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_2':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.12, "ic/lat-geod-deg": 0.1,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                            }
                    }
    elif agent_nums == 4:
        dict_init = {'flag':{'init':{'render': render}},
                    'red':{
                        'red_0':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.01,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_1':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.05,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_2':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.1,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_3':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.15,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        },
                    'blue':{
                        'blue_0':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.01,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_1':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.05,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_2':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.1,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_3':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.15,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                            }
                    }
    elif agent_nums == 5:
        dict_init = {'flag':{'init':{'render': render}},
                    'red':{
                        'red_0':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.01,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_1':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.05,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_2':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.1,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_3':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.15,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        'red_4':
                                {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05, "ic/lat-geod-deg": 0.2,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90
                                },
                        },
                    'blue':{
                        'blue_0':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.01,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_1':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.05,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_2':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.1,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_3':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.15,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                        'blue_4':
                            {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05, "ic/lat-geod-deg": 0.2,
                            "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0, "ic/r-rad_sec": 0,
                            "ic/roc-fpm": 0, "ic/psi-true-deg": -90
                            },
                            }
                    }
    elif agent_nums == 6:
        dict_init = {'flag':{'init':{'render': render}},
                    'red': {
                        'red_0': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05,
                                "ic/lat-geod-deg": 0.01,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                        'red_1': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05,
                                "ic/lat-geod-deg": 0.05,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                        'red_2': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05,
                                "ic/lat-geod-deg": 0.1,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                        'red_3': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05,
                                "ic/lat-geod-deg": 0.15,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                        'red_4': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05,
                                "ic/lat-geod-deg": 0.2,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90},
                        'red_5': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": -0.05,
                                "ic/lat-geod-deg": 0.25,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": 90}
                    },
                    'blue': {
                        'blue_0': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg":  0.05,
                                "ic/lat-geod-deg": 0.01,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                        'blue_1': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg":  0.05,
                                "ic/lat-geod-deg": 0.05,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                        'blue_2': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg":  0.05,
                                "ic/lat-geod-deg": 0.1,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                        'blue_3': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg":  0.05,
                                "ic/lat-geod-deg": 0.15,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                        'blue_4': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg":  0.05,
                                "ic/lat-geod-deg": 0.2,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": -90},
                        'blue_5': {"ic/h-sl-ft": 28000, "ic/terrain-elevation-ft": 1e-08, "ic/long-gc-deg": 0.05,
                                "ic/lat-geod-deg": 0.25,
                                "ic/u-fps": 590.73, "ic/v-fps": 0, "ic/w-fps": 0, "ic/p-rad_sec": 0, "ic/q-rad_sec": 0,
                                "ic/r-rad_sec": 0,
                                "ic/roc-fpm": 0, "ic/psi-true-deg": -90}
                    }}
    if reset:
        dict_init['flag'] = {'reset':{}}
    return dict_init
