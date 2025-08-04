// Main JavaScript for TMDB Movie Recommendation System

// Global variables
let currentUser = null;
let currentAlgorithm = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎬 TMDB Movie Recommendation System initialized');
    
    // Add smooth scrolling to all links
    addSmoothScrolling();
    
    // Add loading states to forms
    addLoadingStates();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Add fade-in animations
    addFadeInAnimations();
});

// Add smooth scrolling to all anchor links
function addSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Add loading states to forms
function addLoadingStates() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function() {
            const submitButton = this.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
            }
        });
    });
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Add fade-in animations to elements
function addFadeInAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    const elements = document.querySelectorAll('.card, .stat-item, .feature-icon');
    elements.forEach(el => observer.observe(el));
}

// Utility function to show notifications
function showNotification(message, type = 'info') {
    const alertClass = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    }[type] || 'alert-info';
    
    const notification = document.createElement('div');
    notification.className = `alert ${alertClass} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Utility function to format numbers
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Utility function to format currency
function formatCurrency(amount) {
    if (amount === null || amount === undefined || amount === 0) {
        return 'Unknown';
    }
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

// Utility function to truncate text
function truncateText(text, maxLength = 100) {
    if (text && text.length > maxLength) {
        return text.substring(0, maxLength) + '...';
    }
    return text;
}

// Utility function to debounce function calls
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Search functionality
function initializeSearch() {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) return;
    
    const debouncedSearch = debounce(function(query) {
        performSearch(query);
    }, 300);
    
    searchInput.addEventListener('input', function() {
        const query = this.value.trim();
        if (query.length >= 2) {
            debouncedSearch(query);
        } else {
            clearSearchResults();
        }
    });
}

// Perform search
function performSearch(query) {
    fetch(`/api/movie_search?q=${encodeURIComponent(query)}`)
        .then(response => response.json())
        .then(data => {
            displaySearchResults(data);
        })
        .catch(error => {
            console.error('Search error:', error);
            showNotification('Search failed. Please try again.', 'error');
        });
}

// Display search results
function displaySearchResults(results) {
    const resultsContainer = document.getElementById('searchResults');
    if (!resultsContainer) return;
    
    if (results.length === 0) {
        resultsContainer.innerHTML = '<p class="text-muted">No movies found.</p>';
        return;
    }
    
    let html = '';
    results.forEach(movie => {
        html += `
            <div class="search-result-item p-2 border-bottom">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="mb-1">${movie.title}</h6>
                        <p class="text-muted small mb-1">${movie.genres}</p>
                        <p class="text-muted small mb-0">${truncateText(movie.overview, 80)}</p>
                    </div>
                    <span class="badge bg-warning text-dark">
                        <i class="fas fa-star me-1"></i>${movie.vote_average}
                    </span>
                </div>
            </div>
        `;
    });
    
    resultsContainer.innerHTML = html;
}

// Clear search results
function clearSearchResults() {
    const resultsContainer = document.getElementById('searchResults');
    if (resultsContainer) {
        resultsContainer.innerHTML = '';
    }
}

// Movie details modal functionality
function showMovieDetailsModal(movieId) {
    const modal = new bootstrap.Modal(document.getElementById('movieModal'));
    
    // Show loading state
    const modalBody = document.getElementById('movieModalBody');
    modalBody.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="mt-3 text-muted">Loading movie details...</p>
        </div>
    `;
    
    modal.show();
    
    // Load movie details
    fetch(`/api/movie_details/${movieId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                modalBody.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        ${data.error}
                    </div>
                `;
                return;
            }
            
            displayMovieDetails(data, modalBody);
        })
        .catch(error => {
            console.error('Error loading movie details:', error);
            modalBody.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    Failed to load movie details.
                </div>
            `;
        });
}

// Display movie details in modal
function displayMovieDetails(movie, modalBody) {
    modalBody.innerHTML = `
        <div class="row">
            <div class="col-md-8">
                <h6>Overview</h6>
                <p>${movie.overview || 'No overview available.'}</p>
                
                <h6>Details</h6>
                <ul class="list-unstyled">
                    <li><strong>Genres:</strong> ${movie.genres}</li>
                    <li><strong>Rating:</strong> ${movie.vote_average}/10</li>
                    <li><strong>Release Date:</strong> ${movie.release_date || 'Unknown'}</li>
                    <li><strong>Runtime:</strong> ${movie.runtime || 'Unknown'} minutes</li>
                    <li><strong>Budget:</strong> ${formatCurrency(movie.budget)}</li>
                    <li><strong>Revenue:</strong> ${formatCurrency(movie.revenue)}</li>
                </ul>
                
                ${movie.tagline ? `<h6>Tagline</h6><p class="text-muted">${movie.tagline}</p>` : ''}
            </div>
            <div class="col-md-4">
                <h6>Similar Movies</h6>
                ${movie.similar_movies && movie.similar_movies.length > 0 ? 
                    movie.similar_movies.map(similar => `
                        <div class="card mb-2">
                            <div class="card-body p-2">
                                <h6 class="card-title small">${similar.title}</h6>
                                <p class="card-text small text-muted">Similarity: ${similar.similarity.toFixed(2)}</p>
                            </div>
                        </div>
                    `).join('') : 
                    '<p class="text-muted small">No similar movies found.</p>'
                }
            </div>
        </div>
    `;
}

// Export functions for use in other scripts
window.TMDBApp = {
    showNotification,
    formatNumber,
    formatCurrency,
    truncateText,
    showMovieDetailsModal,
    initializeSearch
}; 