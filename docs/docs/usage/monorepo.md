# Work with an Existing Mono Repo using RA.Aid

RA.Aid is good at handling complex monorepo projects. For example, if you you have an app set up like this:

```
app/
web/
backend/
```

You can run RA.Aid at the top level and give it commands like this:

```
ra-aid -m "Update the user form to support birthdates. Be sure to also update the DB model and migration scripts."
```

RA.Aid will proceed to work on multiple high-level components of your application.