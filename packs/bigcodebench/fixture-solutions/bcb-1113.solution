import csv
import collections

def task_func(csv_file, emp_prefix='EMP$$'):
    """
    Count the number of records for each employee in a CSV file.
    
    Parameters:
    csv_file (str): The path to the CSV file. This parameter is mandatory.
    emp_prefix (str): The prefix of the employee IDs. Default is 'EMP$$'.
    
    Returns:
    dict: A dictionary with the count of records for each employee.
    
    Requirements:
    - csv
    - collections
    
    Example:
    >>> counts = task_func('/path/to/file.csv')
    >>> print(counts)
    {'EMP$$001': 5, 'EMP$$002': 3}
    """
    counter = collections.Counter()
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0].startswith(emp_prefix):
                    counter[row[0]] += 1
    except FileNotFoundError:
        return {"error": f"The file {csv_file} was not found."}
    except Exception as e:
        return {"error": str(e)}
    
    return dict(counter)
