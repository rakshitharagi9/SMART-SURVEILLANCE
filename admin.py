import streamlit as st
import sqlite3
import hashlib

# Helpers
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def fetch_all_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT username, email, registration_date FROM users")
    users = c.fetchall()
    conn.close()
    return users

def delete_user(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

# Admin creds
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = hash_password("admin123")

def main():
    st.set_page_config(page_title="Admin Panel", layout="wide")
    st.title("🛠️ Admin Panel – User Management")

    # — Authentication —
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Log in"):
            if u == ADMIN_USERNAME and hash_password(p) == ADMIN_PASSWORD_HASH:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        return

    st.sidebar.button("Log out", on_click=lambda: st.session_state.update(admin_authenticated=False))

    st.subheader("Registered Users")
    users = fetch_all_users()
    if not users:
        st.info("No users found.")
        return

    for username, email, reg_date in users:
        cols = st.columns([3, 1])
        # 1) Details
        cols[0].markdown(f"**{username}**  \n📧 {email}  \n📅 {reg_date}")
        
        # 2) Delete button
        if st.session_state.get(f"delete_{username}_confirm", False):
            st.warning(f"Are you sure you want to **delete** user `{username}`?")
            c1, c2 = st.columns(2)
            if c1.button("Yes, delete", key=f"confirm_yes_{username}"):
                delete_user(username)
                st.success(f"Deleted user `{username}`")
                st.session_state.pop(f"delete_{username}_confirm")
                st.rerun()
            if c2.button("No, cancel", key=f"confirm_no_{username}"):
                st.session_state.pop(f"delete_{username}_confirm")
        else:
            # Show the delete button
            if cols[1].button("Delete", key=f"delete_{username}"):
                st.session_state[f"delete_{username}_confirm"] = True

        st.markdown("---")

if __name__ == "__main__":
    main()
