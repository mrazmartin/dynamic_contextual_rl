In the example is a simple example how to use dynamic CARL with dmc control

The structure is pretty simple, you just choose this environment and decide
- which contexts to use per episode initialization
- which dynamic functions to use in each episode (for now only self callable time scheduling ones)
- the rendering mode you prefer (what will get called upon .render() )

And optionally also
- task_kwargs which will past forward to the task initialization
- environment_kwargs which will be past forward to the env initialization
    -> (usually just ask for flat observations)
- specific env kwargs, e.g. HP for dynamic context for the dynamic CARL wrapper

This is a self standing dynamic contextual environment which you can provide to PPO and it will do everything else for you (BUT LOGGING)