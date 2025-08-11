from django.http import JsonResponse
from django.utils.deprecation import MiddlewareMixin
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from .models import User
import json

class RoleBasedAccessMiddleware(MiddlewareMixin):
    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)
        
    def process_request(self, request):
        print(f"DEBUG: Processing request for path: {request.path}")
        
        # Skip authentication for public endpoints
        public_paths = [
            '/api/auth/register/',
            '/api/auth/login/',
            '/api/auth/token/refresh/',
        ]
        
        # Skip for Django admin and static files
        if (request.path.startswith('/admin/') or 
            request.path.startswith('/static/') or
            request.path in public_paths):
            print(f"DEBUG: Skipping authentication for path: {request.path}")
            return None
            
        # Protected endpoints that require member or admin role
        protected_paths = [
            '/api/market-stats/',
            '/api/news/',
            '/api/news-sentiment/',
            '/api/iterative-trading-signals/',
            '/api/signal-performance/',
            '/api/llm-stream/',
        ]
        
        # Admin only endpoints
        admin_paths = [
            '/api/admin/',
        ]
        
        # Check if this is a protected path
        is_protected = any(request.path.startswith(path) for path in protected_paths)
        is_admin_only = any(request.path.startswith(path) for path in admin_paths)
        
        if not is_protected and not is_admin_only:
            return None
            
        # Try to authenticate using JWT
        jwt_authenticator = JWTAuthentication()
        try:
            auth_header = request.META.get('HTTP_AUTHORIZATION', '')
            print(f"DEBUG: Authorization header: {auth_header[:50]}..." if auth_header else "DEBUG: No Authorization header found")
            
            auth_result = jwt_authenticator.authenticate(request)
            if auth_result is None:
                print(f"DEBUG: JWT authentication failed for {request.path} - no valid token found")
                # Try session authentication as fallback
                if hasattr(request, 'user') and request.user.is_authenticated:
                    user = request.user
                    print(f"DEBUG: Using session authentication for user {user.email}")
                else:
                    print(f"DEBUG: No session user found, request.user: {getattr(request, 'user', 'None')}")
                    return JsonResponse({
                        'error': 'Authentication required',
                        'code': 'AUTH_REQUIRED'
                    }, status=401)
            else:
                user, token = auth_result
                request.user = user
                print(f"DEBUG: JWT authentication successful for user {user.email}")
            
        except (InvalidToken, TokenError):
            return JsonResponse({
                'error': 'Invalid or expired token',
                'code': 'INVALID_TOKEN'
            }, status=401)
        except Exception as e:
            print(f"DEBUG: JWT authentication exception: {e}")
            return JsonResponse({
                'error': 'Authentication failed',
                'code': 'AUTH_FAILED'
            }, status=401)
        
        # Check role-based access
        if is_admin_only and not user.is_admin_user():
            print(f"DEBUG: Admin access denied for user {user.email}, is_admin: {user.is_admin_user()}")
            return JsonResponse({
                'error': 'Admin access required',
                'code': 'ADMIN_REQUIRED'
            }, status=403)
        
        if is_protected and not user.can_access_dashboard():
            print(f"DEBUG: Dashboard access denied for user {user.email}")
            print(f"DEBUG: Role: {user.role}, is_superuser: {user.is_superuser}")
            print(f"DEBUG: can_access_dashboard: {user.can_access_dashboard()}")
            return JsonResponse({
                'error': 'Member access required. Please contact admin for membership.',
                'code': 'MEMBER_REQUIRED',
                'user_role': user.role
            }, status=403)
        
        print(f"DEBUG: Access granted for user {user.email} to path {request.path}")
        
        return None