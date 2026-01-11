# routes/web.py - Web routes for serving frontend templates
"""
Web Layer - Routes for serving HTML templates and static files.
Provides the main web interface for the Crop Disease Prediction System.
"""

from flask import Blueprint, render_template, current_app, session, request, redirect, url_for

web_bp = Blueprint('web', __name__)

@web_bp.route('/')
def index():
    """
    Main application page.

    Returns:
        HTML: Main application interface
    """
    return render_template('index.html')

@web_bp.route('/predict')
def predict_page():
    """
    Disease prediction page.

    Returns:
        HTML: Prediction interface
    """
    return render_template('predict.html')

@web_bp.route('/results/<session_id>')
def results_page(session_id):
    """
    Prediction results page.

    Args:
        session_id: Prediction session identifier

    Returns:
        HTML: Results display page
    """
    return render_template('results.html', session_id=session_id)

@web_bp.route('/about')
def about():
    """
    About page with system information.

    Returns:
        HTML: About page
    """
    return render_template('about.html')

@web_bp.route('/help')
def help_page():
    """
    Help and documentation page.

    Returns:
        HTML: Help page
    """
    return render_template('help.html')

@web_bp.route('/lang/<lang_code>')
def set_language(lang_code):
    """
    Set the application language.

    Args:
        lang_code: Language code ('en' or 'hi')

    Returns:
        Redirect: Back to the referring page
    """
    if lang_code in ['en', 'hi']:
        session['lang'] = lang_code

    # Get the referrer or default to home page
    referrer = request.referrer
    if referrer and current_app.url_for('web.index', _external=True) in referrer:
        return redirect(referrer)
    else:
        return redirect(url_for('web.index'))