import pandas as pd
import random

# Function to generate synthetic training data
def generate_training_data(num_samples=1000):
    case_templates = [
        "The user is unable to {action} due to {reason}.",
        "{system} throws an error when trying to {action}.",
        "Performance is degraded when {action} occurs.",
        "{device} is not functioning properly after {event}.",
        "The application crashes while {action}.",
        "{user_type} cannot access {system} due to {reason}.",
        "{process} fails intermittently when {action} is performed.",
    ]

    resolution_templates = [
        "Please {solution_action} and ensure {solution_check}.",
        "Advise the user to {solution_action}.",
        "Verify that {system_check} and {config_change}.",
        "Ensure that {process_step} is completed.",
        "Perform a {system_action} and test again.",
        "Apply {configuration} and monitor for changes.",
        "Upgrade the system to {version} and recheck."
    ]

    variables = {
        "action": ["log in", "submit a form", "upload a file", "reset the password", "start the service"],
        "reason": ["network issues", "incorrect credentials", "insufficient permissions", "a timeout", "a configuration error"],
        "system": ["the application", "the database", "the web server", "the mobile app"],
        "device": ["the printer", "the scanner", "the router", "the workstation"],
        "event": ["a firmware update", "a software installation", "a configuration change"],
        "user_type": ["Admin users", "Guest users", "Regular employees"],
        "process": ["the authentication", "the deployment", "the synchronization"],
        "solution_action": ["restart the system", "update the drivers", "clear the cache"],
        "solution_check": ["the configurations are correct", "the user credentials are valid"],
        "system_check": ["the network connection is stable", "the database is accessible"],
        "config_change": ["reset the user permissions", "reconfigure the application settings"],
        "process_step": ["a backup is taken", "all dependencies are installed"],
        "system_action": ["reboot", "rollback the update"],
        "configuration": ["the latest patch", "default settings"],
        "version": ["version 2.1", "the most recent release"]
    }

    data = []
    for _ in range(num_samples):
        case = random.choice(case_templates).format(
            **{key: random.choice(values) for key, values in variables.items()}
        )
        resolution = random.choice(resolution_templates).format(
            **{key: random.choice(values) for key, values in variables.items()}
        )
        data.append({"case_description": case, "resolution": resolution})

    return pd.DataFrame(data)

# Generate 1000 lines of data
training_data = generate_training_data(1000)

# Save to CSV for use in the script
file_path = "support_cases.csv"
training_data.to_csv(file_path, index=False)
file_path
