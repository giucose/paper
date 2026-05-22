
def get_repository_id(year, link):
    repository_id = str(year)  + "_" + link.replace("https://github.com/", "").replace("/", "_")
    return repository_id