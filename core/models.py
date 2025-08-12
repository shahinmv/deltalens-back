from django.db import models
from django.contrib.auth.models import AbstractUser
import uuid

class MarketStats(models.Model):
    market_cap = models.FloatField()
    market_cap_change_24h = models.FloatField()
    market_cap_change_percentage_24h = models.FloatField()
    btc_dominance = models.FloatField()
    btc_dominance_change_24h = models.FloatField()
    volume_24h = models.FloatField()
    volume_24h_change = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Market Stats"

    def __str__(self):
        return f"Market Stats - {self.updated_at}"

class User(AbstractUser):
    ROLE_CHOICES = [
        ('non_member', 'Non-Member'),
        ('member', 'Member'),
        ('admin', 'Admin'),
    ]
    
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='non_member'
    )
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']
    
    def __str__(self):
        return self.email
    
    def can_access_dashboard(self):
        return self.role in ['member', 'admin'] or self.is_superuser
    
    def is_admin_user(self):
        return self.role == 'admin' or self.is_superuser
    
    def save(self, *args, **kwargs):
        # Automatically assign admin role to superusers
        if self.is_superuser and self.role == 'non_member':
            self.role = 'admin'
        super().save(*args, **kwargs)


class ConversationSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversation_sessions')
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.user.email} - {self.title or f'Session {self.id}'}"

    def generate_title(self):
        """Generate a title based on the first user message"""
        first_message = self.messages.filter(is_user=True).first()
        if first_message:
            # Take first 50 characters of the first user message
            title = first_message.content[:50]
            if len(first_message.content) > 50:
                title += "..."
            self.title = title
            self.save(update_fields=['title'])
        return self.title


class ConversationMessage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ConversationSession, on_delete=models.CASCADE, related_name='messages')
    content = models.TextField()
    is_user = models.BooleanField()
    reasoning = models.TextField(blank=True)  # For AI reasoning/thinking
    tool_calls = models.JSONField(blank=True, null=True)  # Store tool calls made
    tool_responses = models.JSONField(blank=True, null=True)  # Store tool responses
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{'User' if self.is_user else 'AI'}: {self.content[:50]}..."
