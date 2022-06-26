@component(packages_to_install=[],
           output_component_file="data_validation_component.yaml",)
def data_validation_component(text: str,
                              project_dict: dict) -> bool:

    import re
    
    result = bool(re.search('hello', text))
    
    PROJECT_ID = project_dict['PROJECT_ID']
    GCS_BUCKET = project_dict['GCS_BUCKET']
    REGION = project_dict['REGION']
    
    print(PROJECT_ID, GCS_BUCKET, REGION)
    
    return result