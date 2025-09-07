
import signal
import sys
from SingleInstanceEvolution import start_single_instance_evolution

# =============================================================================
# MAIN EXECUTION MODULE
# =============================================================================

def main():
    """Initialize and start the evolutionary training process"""
    try:
        start_single_instance_evolution()
    except KeyboardInterrupt:
        print("\nTraining terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()
