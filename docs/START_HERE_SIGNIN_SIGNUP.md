# Start here: ReliaGraph sign-in and sign-up

This package keeps **Sign in** and **Sign up** visible as tabs on the initial authentication page. The authentication page is visible immediately while the local server checks session status. On an empty workspace, the app automatically selects **Sign up**; if accounts already exist, it selects **Sign in**, and the **Sign up** tab remains available.

## Start on Windows

1. Extract the entire ZIP to a writable folder.
2. Double-click `run_reliagraph_windows.bat`.
3. Keep the terminal window open and open `http://127.0.0.1:8765`.
4. Click **Sign up**, enter a username and password of at least 12 characters, confirm the password, and create your account.
5. On later visits, use **Sign in**.

Do not open `webapp/static/index.html` directly as a `file://` URL. The page requires the local API started by the batch file for account creation and sign-in.

Accounts authenticate separately, but this prototype shares the same local documents and knowledge graph across accounts; it is not yet isolated multi-tenant storage.
